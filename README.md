# `Qwen3.8-27B` LoRA & QLoRA Fine-Tuning (`v5p-16`) and Multi-LoRA Serving (`v7x`) on Cloud TPU

End-to-end **parameter-efficient fine-tuning (`LoRA`, `INT8 QLoRA`, `NF4 QLoRA`)** and **static / dynamic / `FP8`-base multi-adapter serving** stack for **`Qwen/Qwen3.8-27B`** (`26.98B` hybrid `48` Gated-DeltaNet + `16` gated full-attention decoder) on **Google Cloud TPU `v5p-16` (`2x2x4`)** with **TorchTitan + TorchTPU `0.2.0`** and **Cloud TPU `v7x`** with **`vllm-torchtpu`**.

Includes a verified **end-to-end train-to-serve loop**: the `r=16` `PEFT` adapter (`79.69M` trainable parameters, `159.4 MB` safetensors) trained on `yahma/alpaca-cleaned` on TPU `v5p-16` is exported in Hugging Face `PEFT` format and dynamically hot-loaded via `POST /v1/load_lora_adapter` into `vllm-torchtpu` on TPU `v7x` alongside community adapters (`r=16` and `r=32`; the `r=128` `taste` adapter loads without effect in this model class, see [Known Limitations](#known-limitations)).

---

## 1. Fine-Tuning Benchmark Results (`TPU v5p-16`, `2x2x4`, `131,072` tok/step, `150` Steps)

All three recipes train identical `r=16, alpha=32` adapters (`79,691,776` trainable parameters = `0.295%` of `26.98B` across `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) over `19.66M` packed SFT tokens (`seq_len=4096`, `microbatch=2/chip`, `SimpleFSDP` ZeRO-3 over `16` chips, `FullAC` activation checkpointing, `DEFER_AND_FUSE` eager execution). Steady state averages steps `11..150`. Throughput is tok/s/chip; `MFU` uses LoRA model FLOPs; HBM sources are defined in [Methodology Notes](#methodology-notes).

| Recipe & Frozen Base Format | Quantized Params (`Bits/Param`) | Allocated HBM / Chip After Load (+Quant) | Peak Allocated HBM / Chip (`torch_tpu`) | Steady Step Time | Throughput / Chip (`16`-Chip Total) | `MFU` (LoRA FLOPs) / `HFU` (`+FullAC`) | Loss (`Step 1 -> Last-10 Mean`) | Code Anchor & Result Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`qwen38_27b_lora_bf16`** *(Frozen `BF16` Base)* | `0` (`16.0` bit) | `4.549 GiB` | `32.415 GiB` *(30-step run: `32.414 GiB`; `tpu-info` max `28.65 GiB`, steps `11–30`)* | `9.772 s` | **`838.9` tok/s** (`13,422` tok/s) | `19.26%` / `28.77%` | `1.1522 -> 0.8498` | [`recipes#L678`](./finetuning/scripts/lora_tpu_recipes.py#L678-L681) · [`stats`](./finetuning/results/lora_bf16_run_stats.json) · [`hbm`](./finetuning/results/lora_bf16_hbm_telemetry.json) |
| **`qwen38_27b_qlora_int8`** *(Frozen `INT8` Channelwise Base)* | `24.35B` (`8.0` bit) | **`3.143 GiB`** *(`-30.9%`)* | **`30.906 GiB`** *(`-4.7%`; `tpu-info` max `30.19 GiB`, steps `11–150`)* | **`9.357 s`** *(`-4.2%`)* | **`876.0` tok/s** (**`14,016` tok/s**, `+4.4%`, within run-to-run variation; see finding 1) | **`20.11%` / `30.05%`** | `1.1555 -> 0.8500` | [`recipes#L198`](./finetuning/scripts/lora_tpu_recipes.py#L198-L246) · [`stats`](./finetuning/results/qlora_int8_run_stats.json) · [`hbm`](./finetuning/results/qlora_int8_hbm_telemetry.json) |
| **`qwen38_27b_qlora_nf4`** *(Frozen `NF4` Block-64 Double-Quant Base)* | `24.35B` (`4.125` bit) | **`2.474 GiB`** *(`-45.6%`)* | `42.767 GiB` *(`+31.9%`; `tpu-info` max `41.98 GiB`, steps `11–150`; FP32 select-tree decode)* | `13.226 s` *(`+35.3%`)* | `619.6` tok/s (`9,914` tok/s, `-26.1%`) | `14.22%` / `21.25%` | `1.1221 -> 0.8543` | [`recipes#L301`](./finetuning/scripts/lora_tpu_recipes.py#L301-L359) · [`stats`](./finetuning/results/qlora_nf4_run_stats.json) · [`hbm`](./finetuning/results/qlora_nf4_hbm_telemetry.json) |

### Key Fine-Tuning Findings
1. **`INT8 QLoRA` vs `BF16 LoRA`: same loss, smaller resident weights, throughput on par:** Quantizing the `592` frozen decoder-layer `nn.Linear` weights (`24.35B` parameters, `0.0104` Frobenius relative error) to `int8` and overriding `SimpleFSDP` parametrization `param_dtype=None` ([`lora_tpu_recipes.py#L237-L240`](./finetuning/scripts/lora_tpu_recipes.py#L237-L240)) all-gathers `1 B/param` `int8` shards over ICI instead of `2 B/param` `bf16` shards. Loss is unchanged (`0.8500` vs `0.8498`, last-10 mean). Allocated HBM after load drops **`30.9%` (`4.549 -> 3.143 GiB/chip`)**, while peak allocated HBM drops `4.7%` (`32.415 -> 30.906 GiB/chip`): at `16`-way FSDP the frozen BF16 base is about `3.1 GiB/chip`, so activations and gathered layers dominate the peak. Over steps `11–30`, INT8 ran at `872.5` tok/s/chip and two BF16 runs on different nodes at `825.0` and `885.4` tok/s/chip, so INT8 throughput is on par with BF16 within run-to-run variation.
2. **`NF4 QLoRA` (`4.125 bit/param`): smallest resident weights, lower throughput, higher peak HBM:** Packs `64`-weight blocks using the 16-level NormalFloat codebook plus `uint8` double-quantized block `absmax` into a single `FSDP`-sharded `uint8` tensor (`[out, in/2 + in/64]`). Allocated HBM after load drops `45.6%` (`2.474 GiB/chip`) and loss converges to `0.8543` (`+0.0045` vs `BF16`, last-10 mean). Dequantization is plain PyTorch (an FP32 15-way select tree followed by a BF16 matmul), with `26.1%` lower tok/s/chip (`619.6` vs `838.9`; `25–30%` lower than the two BF16 runs over steps `11–30`) and `31.9%` higher peak allocated HBM (`42.767` vs `32.415 GiB/chip`; `tpu-info` max `41.98 GiB`).
3. **Direct Sharded Safetensors Loader:** Each rank slices only its local `DTensor` offset directly via `safe_open` ([`lora_tpu_recipes.py#L401-L452`](./finetuning/scripts/lora_tpu_recipes.py#L401-L452)), avoiding distributed DCP planner stalls on fused Gated-DeltaNet `in_proj_qkv` / `conv1d` weights. Load time is `2.1–2.5 s` on hosts that have already read the checkpoint and `139.2 s` on freshly provisioned nodes.
4. **Loss curves on the same data and seed:** `INT8` tracks `BF16` with a mean absolute difference of `0.0002` and a maximum of `0.0033` over all `150` steps. `NF4` starts from a different step-1 loss (`1.1221` vs `1.1522`) because its base weights carry about `9%` relative error (`0.0927` Frobenius), and ends `+0.0045` above `BF16` (last-10 mean). Per-step losses are in [`summary.json`](./finetuning/results/summary.json) (`loss_curves`); agreement statistics are under `notes`.
5. **HBM by `tpu-info` over the same steps (`11–30`):** max / p95 / median `28.65 / 27.25 / 22.02 GiB` for BF16 (30-step run with the same recipe, launched with the batch Job manifest; peak allocated `32.414 GiB/chip`), `28.23 / 26.51 / 21.20 GiB` for INT8 and `41.44 / 36.68 / 25.07 GiB` for NF4 ([`lora_bf16_hbm_telemetry.json`](./finetuning/results/lora_bf16_hbm_telemetry.json), [`qlora_int8_hbm_telemetry.json`](./finetuning/results/qlora_int8_hbm_telemetry.json), [`qlora_nf4_hbm_telemetry.json`](./finetuning/results/qlora_nf4_hbm_telemetry.json)).

### Cross-Check Against the Flint MoE Study ([`rsi-flint-tpu-lora-qlora-finetuning`](https://github.com/ankdeshm/rsi-flint-tpu-lora-qlora-finetuning))
Same method and harness family (TorchTitan TPU lane, `SimpleFSDP` over `16` `v5p` chips, `FullAC`, `INT8` per-channel and `NF4` block-64 frozen bases all-gathered without upcast, `tpu-info` telemetry, `PEFT` export, `alpaca-cleaned` at `2 x 4096` tokens/chip, `r=16, alpha=32`, `150` steps), applied to a different model class: Flint is a `39.8B` sparse MoE (`2.5B` active parameters per token), while `Qwen3.8-27B` is dense (all `27B` parameters active).
- **Agree:** `INT8` matches `BF16` loss in both studies; `NF4` has the smallest resident footprint but the highest peak HBM and the lowest throughput in both.
- **Differ:** on `Qwen3.8-27B`, `INT8` throughput is on par with `BF16` (within run-to-run variation), while Flint reports `INT8` at `95.3%` of `BF16`; the `NF4` throughput penalty is smaller here (`-26%` vs `BF16`) than on Flint (`-67%`), consistent with the largest dequantized tensor being one `89M`-parameter `Linear` here versus `256`-expert weight banks there.
- Absolute tok/s, MFU and HBM numbers are not comparable across the two studies (different active compute per token and different MFU conventions).

---

## 2. Multi-LoRA & QLoRA-Style Serving Results (`TPU v7x`, `1` TensorCore `TP=1` Worker, `1024 in / 256 out`)

Measured on a single `TPU v7x` TensorCore (`TP=1`, `94.74 GiB` HBM visible to the vLLM device; one worker on one host) using `vllm-torchtpu-bench:v0.32.0` with the two `vllm-torchtpu` changes listed under [Modifications](#modifications). Workload: random prompts, `1024` input / `256` output tokens, streaming OpenAI API; output tok/s is aggregate across concurrent requests. Every benchmark point ran with `0` failed requests (`100%` pass rate).

| Serving Phase & Base Checkpoint | Active Request Routing (`max_loras=2`, `max_rank=128`) | `c=1` Output (`TPOT p50`) | `c=8` Output (`TTFT p50` / `TPOT p50`) | `c=32` Output (`TTFT p50` / `TPOT p50`) | Artifact |
|---|---|---:|---:|---:|---|
| **`p1`: `BF16` Base (No LoRA Baseline)** | `Qwen/Qwen3.8-27B` | **`52.21` tok/s** (`18.60 ms`) | **`344.50` tok/s** (`573.5 ms` / `21.13 ms`) | **`847.00` tok/s** (`670.5 ms` / `34.66 ms`) | [`p1 JSON`](./serving/results/p1_base_bf16_summary.json) |
| **`p2`: Static Multi-LoRA (`BF16` Base)** | Single adapter: `taste` (`r=128`): serving cost of an enabled `r=128` LoRA slot; the adapter loads without effect ([Known Limitations](#known-limitations)) | `36.40` tok/s (`26.23 ms`) | `238.06` tok/s (`836.0 ms` / `29.84 ms`) | `637.74` tok/s (`855.5 ms` / `45.61 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p2`: Static Multi-LoRA (`BF16` Base)** | Mixed batch: `base` + `taste` (`r=128`, enabled slot without effect) + `taiwan` (`r=16`) | — | `249.29` tok/s *(2-LoRA)* (`806.3 ms` / `28.99 ms`) | **`655.65` tok/s** (`855.9 ms` / `45.26 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — Train-to-Serve)** | **`tpu_sft` (`r=16`, TPU `v5p-16` TorchTitan adapter)** via `/v1/load_lora_adapter` (`0.16 s`) | `36.31` tok/s (`26.21 ms`) | `238.01` tok/s (`849.5 ms` / `29.64 ms`) | **`639.82` tok/s** (`855.5 ms` / `45.24 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — No Swap)** | `taste` (enabled slot without effect) + `taiwan` (`2` adapters in `2` HBM slots) | — | `243.55` tok/s (`838.1 ms` / `29.42 ms`) | **`654.09` tok/s** (`855.5 ms` / `45.25 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — Live Eviction/Swap)** | **`4` concurrent adapters (`tpu_sft + taste + taiwan + jev`) -> `2` HBM slots (`max_cpu_loras=4`)**; `taste` slot without effect | — | `154.35` tok/s (`2,090 ms` / `28.84 ms`) | **`317.55` tok/s** (`9,833 ms` / `35.59 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p4`: `FP8`-Base QLoRA-Style Serving** | `Qwen/Qwen3.8-27B-FP8` Base (`--enable-lora` active, unadapted requests) | `49.55` tok/s (`19.19 ms`) | `319.43` tok/s (`587.7 ms` / `21.26 ms`) | **`854.93` tok/s** (`605.1 ms` / `33.36 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p4`: `FP8`-Base QLoRA-Style Multi-LoRA** | Mixed `taste` (`r=128`, enabled slot without effect) + `taiwan` (`r=16`) `BF16` adapters on `FP8` Base | — | `353.22` tok/s (`523.2 ms` / `20.73 ms`) | `889.78` tok/s (`605.0 ms` / `33.35 ms`) | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |

### Key Serving Findings
1. **Static and dynamic adapter loading:** adapters register at startup (`--lora-modules`) or at runtime via `/v1/load_lora_adapter` (`0.16–0.85 s` per adapter) and `/v1/unload_lora_adapter` (`<0.01 s`), without a server restart. The TPU-trained `tpu_sft` adapter serves at `639.82` tok/s at `c=32`. LoRA application is verified on `q/k/v/o_proj` and `gate/up/down_proj` (`tpu_sft` and `taiwan` change outputs).
2. **`FP8`-Base + `BF16` LoRA ("QLoRA-Style Serving"):** with LoRA enabled, `FP8`-base requests at `c=32` reach **`854.93` tok/s** (`33.36 ms` TPOT), `+0.9%` vs the `BF16` base with LoRA disabled (`p1`, `847.00` tok/s) and `+33.6%` vs the `BF16` base with LoRA enabled serving `tpu_sft` (`639.82` tok/s). There is no `FP8` reference with LoRA disabled. Footprint: checkpoint `28.75` vs `51.75 GiB`; HBM used after load `33.29` vs `55.33 GiB` (both LoRA-enabled); KV cache `512,722` vs `295,393` tokens.
3. **Cost of enabling LoRA (`BF16` base, `max_lora_rank=128`, `max_loras=2`):** TPOT at `c=1` rises from `18.60` to `26.21 ms` (`tpu_sft`), and `c=32` throughput is `24.5%` lower (`639.82` vs `847.00` tok/s). In `p4`, batches containing only base-model requests run at the same TPOT as adapter batches (`33.36` vs `33.35 ms` at `c=32`): base requests get no LoRA-free fast path once `--enable-lora` is set.
4. **HBM, KV capacity and boot per server configuration** (vLLM `Memory statistics`, one `v7x` TensorCore):

| Server Configuration | Checkpoint | HBM Used After Load + Profile | KV Cache (Tokens) | Engine Init (XLA Compile) |
|---|---:|---:|---:|---:|
| `p1`: `BF16`, LoRA off | `51.75 GiB` | `51.12 GiB` | `336,835` | `262 s` (`186 s`) |
| `p2` / `p3`: `BF16`, LoRA on (`2 x r=128` slots) | `51.75 GiB` | `55.33 GiB` (`+4.21 GiB`) | `295,393` (`-12%`) | `689–694 s` (`622–627 s`) |
| `p4`: `FP8`, LoRA on (`2 x r=128` slots) | `28.75 GiB` | `33.29 GiB` | `512,722` (`+52%` vs `p1`) | `863 s` (`678 s`) |

---

<a id="methodology-notes"></a>
## 3. Methodology Notes

- **Throughput:** tok/s/chip is torchtitan's `tps` (trained tokens per second per chip); the `16`-chip total is `x16`. All deltas between runs are changes in tok/s/chip. Steady-state values average steps `11..150`; the `150`-step wall clock (`1,574.8 / 1,427.4 / 2,007.1 s`) includes step-1 compilation (`116 / 36 / 39 s`).
- **Run-to-run variation:** two BF16 runs on different nodes measured `825.0` and `885.4` tok/s/chip over steps `11–30`.
- **MFU basis:** `MFU` = tok/s/chip x LoRA model FLOPs per token (`4·N_mm + 3·attn_fwd + 6·N_lora` = `105.38 GFLOP`, `N_mm` = `25.62B` matmul parameters; frozen weights get no weight-gradient GEMM) / `459 TFLOP/s` per `v5p` chip. `HFU` adds the `FullAC` forward recompute (`157.43 GFLOP/token`). torchtitan's logged `mfu` uses the full-fine-tune `6N` convention (`29.2%` / `30.5%` / `21.6%` for BF16 / INT8 / NF4) and is listed for reference only. Gated-DeltaNet chunked-scan FLOPs (~1–2%) are not counted.
- **HBM sources:** *allocated after load* and *peak allocated* are `torch_tpu` allocator statistics (right after the checkpoint load + quantization, and the maximum at the end of training); `tpu-info` is the libtpu runtime `hbm_usage` metric sampled every ~2.5 s on the 4 chips of host 0. The `150`-step BF16 run has no `tpu-info` samples; BF16 `tpu-info` comes from a 30-step run with the same recipe (`LORA_FT_STEPS=30`, so its LR schedule differs after warmup). torchtitan's per-step `memory` log field (reserved-bytes peak, `10.87 GiB` in every run) does not track device usage and is not used.
- **Quantized dequantization:** `NF4` dequantization is pure PyTorch (FP32 15-way select tree, then BF16 matmul; no custom or Pallas kernel). `INT8` dequantization is a cast plus per-channel scale in PyTorch with a BF16 matmul.
- **Serving:** one worker (one `v7x` TensorCore, `TP=1`) was measured; output tok/s is aggregate across concurrent requests (`8 / 32 / 128` requests at `c=1 / 8 / 32`); TTFT and TPOT are p50.
- **Quality:** no accuracy benchmark was run; quality is reported as training loss and greedy chat probes ([`p2_p3_p4_multilora_summary.json`](./serving/results/p2_p3_p4_multilora_summary.json)).
- **Harness:** the three `150`-step runs used the interactive dev harness ([`lora-ft-dev-v5p16.yaml`](./finetuning/k8s/lora-ft-dev-v5p16.yaml)); the BF16 30-step run used the batch Job manifest ([`lora-ft-qwen38-27b-v5p16.yaml`](./finetuning/k8s/lora-ft-qwen38-27b-v5p16.yaml)), which completed on all `4` hosts.

<a id="modifications"></a>
## 4. Modifications

**`vllm-torchtpu`** (applied by [`run_vllm_tpu_lora_suite.py`](./serving/scripts/run_vllm_tpu_lora_suite.py#L163-L225) to the container's source before `vllm serve`):
- `vllm_torchtpu/runner/tpu_runner.py`: the TPU model runner initializes the LoRA manager (`load_lora_model`) after loading the model when `--enable-lora` is set.
- `vllm_torchtpu/worker/tpu_worker.py`: the TPU worker implements `add_lora` / `remove_lora` / `pin_lora` / `list_loras` by forwarding to the model runner (used by `--lora-modules` and `/v1/load_lora_adapter`).
- Runtime adapter loading runs with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.

**TorchTitan** (implemented in [`lora_tpu_recipes.py`](./finetuning/scripts/lora_tpu_recipes.py) and applied at import; the TorchTitan source in the image is unchanged):
- A state-dict adapter reads the multimodal Qwen3.8 checkpoint (`model.language_model.*` keys) into the text-only `qwen3_5` model.
- A direct sharded safetensors loader (`LORA_FT_HF_LOADER=direct`, default) reads each rank's own `DTensor` shard.
- Frozen parameters are cast to BF16 after LoRA conversion.
- A post-load hook quantizes the frozen decoder `nn.Linear` weights (`INT8` or `NF4`) and sets `SimpleFSDP` `param_dtype=None` so shards are all-gathered as `int8` / `uint8`.
- A post-training hook exports the adapter in Hugging Face `PEFT` format (BF16) and writes `run_stats.json`.
- `LORA_FT_STEPS` overrides the step count (default `150`).

<a id="known-limitations"></a>
## 5. Supported Configurations & Known Limitations

- **LoRA serving in `vllm-torchtpu`:** `TP=1` and `DP=1`, dense models, no speculative decoding, and LoRA dtype equal to the model dtype.
- **`taste` adapter:** the adapter's tensors use the `model.layers.*` prefix, which this model class (`Qwen3_5ForConditionalGeneration`) does not map, so it loads without effect (greedy outputs match the base model in `p2`, `p3` and `p4`). Its rows show the serving cost of an enabled `r=128` LoRA slot, not adapter behaviour. Renaming the keys to `model.language_model.layers.*`, or serving with `--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'` (whose weight mapper accepts both prefixes), makes the keys match.
- **`FP8` base:** no `FP8` reference with LoRA disabled was measured, so the `FP8`-only gain and the LoRA overhead on an `FP8` base cannot be separated.
- **Gated-DeltaNet LoRA targets:** `jev` targets the GDN `in_proj_qkv` / `in_proj_z` / `out_proj` projections and `lm_head`; it loads and changes outputs, but per-module application on the GDN projections was not verified.
- **LoRA serving cost** was measured with `max_lora_rank=128`; a rank-`16`-only deployment (`max_lora_rank=16`) was not measured.
- **Adapter slots:** with more active adapters than `--max-loras` slots, adapters swap (4 adapters on 2 slots: `317.55` tok/s at `c=32`, TTFT p50 `9.8 s`). Size `--max-loras` to the number of concurrently active adapters.
- **Server boot:** the XLA compile cache is not reused across server restarts in this image; a LoRA-enabled boot takes `11–15 min`.

---

## 6. Repository Layout

- [`docs/QWEN38_27B_TPU_LORA_QLORA_REPORT.md`](./docs/QWEN38_27B_TPU_LORA_QLORA_REPORT.md) — Detailed architecture, FLOP accounting, HBM breakdown, chat completion behavioral probes, and reproduction guide.
- [`finetuning/scripts/`](./finetuning/scripts/) — `lora_tpu_recipes.py` (`BF16` LoRA, `INT8` QLoRA, `NF4` QLoRA + `PEFT` exporter), `prep_sft_data.py`, `launch_ft.sh`, `ft_ctl.sh`, `ft_chain.sh`, `hbm_sampler.sh`, `parse_ft_logs.py`, `make_ft_summary.py`.
- [`finetuning/k8s/`](./finetuning/k8s/) — Indexed 4-host `TPU v5p-16` (`2x2x4`) batch and interactive harness manifests; the batch Job has optional `LORA_FT_STEPS` and `LORA_FT_HBM_SAMPLER` and copies its logs to the models volume on exit.
- [`finetuning/results/`](./finetuning/results/) — `summary.json` (incl. `notes` on conventions), per-recipe `run_stats.json`, `tpu-info` telemetry per run (`lora_bf16_hbm_telemetry.json`, `qlora_int8_hbm_telemetry.json`, `qlora_nf4_hbm_telemetry.json`), `lora_bf16_hbm30_run_stats.json` (BF16 30-step run), and exported `adapter_config.json`.
- [`serving/scripts/`](./serving/scripts/) — `run_vllm_tpu_lora_suite.py` (applies the `vllm-torchtpu` modifications, runs the multi-phase OpenAI streaming benchmark and quality/chat probes) and `tpu_single_host_setup.py` (`LD_PRELOAD` single-host metadata isolator).
- [`serving/plans/`](./serving/plans/) & [`serving/k8s/`](./serving/k8s/) — Benchmark JSON plans (`p1`–`p4`) and `TPU v7x` Kubernetes Job manifest.
- [`serving/results/`](./serving/results/) — Complete `p1`–`p4` latency/throughput JSONs and greedy generation transcripts.
