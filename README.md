# `Qwen3.8-27B` LoRA & QLoRA Fine-Tuning (`v5p-16`) and Multi-LoRA Serving (`v7x`) on Cloud TPU

End-to-end **parameter-efficient fine-tuning (`LoRA`, `INT8 QLoRA`, `NF4 QLoRA`)** and **static / dynamic / `FP8`-base multi-adapter serving** stack for **`Qwen/Qwen3.8-27B`** (`26.98B` hybrid `48` Gated-DeltaNet + `16` gated full-attention decoder) on **Google Cloud TPU `v5p-16` (`2x2x4`)** with **TorchTitan + TorchTPU `0.2.0`** and **Cloud TPU `v7x`** with **`vllm-torchtpu`**.

Includes a verified **end-to-end train-to-serve loop**: the `r=16` `PEFT` adapter (`79.69M` trainable parameters, `159.4 MB` safetensors) trained on `yahma/alpaca-cleaned` on TPU `v5p-16` is exported in Hugging Face `PEFT` format and dynamically hot-loaded via `POST /v1/load_lora_adapter` into `vllm-torchtpu` on TPU `v7x` alongside community adapters up to rank `128`.

---

## 1. Fine-Tuning Benchmark Results (`TPU v5p-16`, `2x2x4`, `131,072` tok/step, `150` Steps)

All three recipes train identical `r=16, alpha=32` adapters (`79,691,776` trainable parameters = `0.295%` of `26.98B` across `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) over `19.66M` packed SFT tokens (`seq_len=4096`, `microbatch=2/chip`, `SimpleFSDP` ZeRO-3 over `16` chips, `FullAC` activation checkpointing, `DEFER_AND_FUSE` eager execution). Steady state averages steps `11..150`.

| Recipe & Frozen Base Format | Quantized Params (`Bits/Param`) | Resting Shard HBM / Chip | Peak Allocated HBM / Chip | Steady Step Time | Throughput / Chip (`16`-Chip Total) | `MFU` (`4N` LoRA) / `HFU` (`+FullAC`) | Loss (`Step 1 -> Last-10 Mean`) | Code Anchor & Result Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`qwen38_27b_lora_bf16`** *(Frozen `BF16` Base)* | `0` (`16.0` bit) | `4.549 GiB` | `32.415 GiB` | `9.772 s` | **`838.9` tok/s** (`13,422` tok/s) | `19.26%` / `28.77%` | `1.1522 -> 0.8498` | [`recipes#L677`](./finetuning/scripts/lora_tpu_recipes.py#L677-L680) · [`stats`](./finetuning/results/lora_bf16_run_stats.json) |
| **`qwen38_27b_qlora_int8`** *(Frozen `INT8` Channelwise Base)* | `24.35B` (`8.0` bit) | **`3.143 GiB`** *(`-30.9%`)* | **`30.906 GiB`** *(tpu-info max `30.19 GiB`)* | **`9.357 s`** *(`-4.2%`)* | **`876.0` tok/s** (**`14,016` tok/s**, **`+4.4%`**) | **`20.11%` / `30.05%`** | `1.1555 -> 0.8500` | [`recipes#L198`](./finetuning/scripts/lora_tpu_recipes.py#L198-L246) · [`stats`](./finetuning/results/qlora_int8_run_stats.json) |
| **`qwen38_27b_qlora_nf4`** *(Frozen `NF4` Block-64 Double-Quant Base)* | `24.35B` (`4.125` bit) | **`2.474 GiB`** *(`-45.6%`)* | `42.767 GiB` *(FP32 select-tree decode)* | `13.226 s` | `619.6` tok/s (`9,914` tok/s) | `14.22%` / `21.25%` | `1.1221 -> 0.8543` | [`recipes#L301`](./finetuning/scripts/lora_tpu_recipes.py#L301-L359) · [`stats`](./finetuning/results/qlora_nf4_run_stats.json) |

### Key Fine-Tuning Findings
1. **`INT8 QLoRA` Beats `BF16 LoRA` on BOTH HBM AND Throughput (`+4.4%` tok/s, `-9.4%` Wall Time):** Quantizing the `592` frozen decoder-layer `nn.Linear` weights (`24.35B` parameters, `0.0104` Frobenius relative error) to `int8` and overriding `SimpleFSDP` parametrization `param_dtype=None` ([`lora_tpu_recipes.py#L237-L240`](./finetuning/scripts/lora_tpu_recipes.py#L237-L240)) all-gathers `1 B/param` `int8` shards over ICI instead of `2 B/param` `bf16` shards—cutting resting shard HBM by **`30.9%` (`4.549 -> 3.143 GiB/chip`)** while reducing `150`-step wall clock from `1,574.8 s` to **`1,427.4 s`** with identical convergence (`0.8500` vs `0.8498`).
2. **`NF4 QLoRA` (`4.125 bit/param`) Cuts Resting Weight HBM by `45.6%` (`2.474 GiB/chip`):** Packs `64`-weight blocks using the 16-level NormalFloat codebook plus `uint8` double-quantized block `absmax` into a single `FSDP`-sharded `uint8` tensor (`[out, in/2 + in/64]`), converging to `0.8543` loss (`+0.0045` vs `BF16`).
3. **Direct Sharded Safetensors Loader (`2.1–2.5 s` Cold Load):** Each rank slices only its local `DTensor` offset directly via `safe_open` ([`lora_tpu_recipes.py#L401-L452`](./finetuning/scripts/lora_tpu_recipes.py#L401-L452)), avoiding distributed DCP planner stalls on fused Gated-DeltaNet `in_proj_qkv` / `conv1d` weights.

---

## 2. Multi-LoRA & QLoRA-Style Serving Results (`TPU v7x`, `1` TensorCore `TP=1` Worker, `1024 in / 256 out`)

Measured on a single `TPU v7x` TensorCore (`TP=1`, `95 GiB` HBM per core; up to `8` independent `TP=1` workers run concurrently on one `4`-chip `v7x` host with zero ICI overhead) using `vllm-torchtpu-bench:v0.32.0` with two lightweight runtime patches ([`run_vllm_tpu_lora_suite.py#L163-L225`](./serving/scripts/run_vllm_tpu_lora_suite.py#L163-L225)). Every benchmark point ran with `0` failed requests (`100%` pass rate).

| Serving Phase & Base Checkpoint | Active Request Routing (`max_loras=2`, `max_rank=128`) | `c=1` Output (`TPOT p50`) | `c=8` Output (`TTFT p50` / `TPOT p50`) | `c=32` Output (`TTFT p50` / `TPOT p50`) | `8`-Worker Single-Host Equivalent (`c=32/worker`) | Artifact |
|---|---|---:|---:|---:|---:|---|
| **`p1`: `BF16` Base (No LoRA Baseline)** | `Qwen/Qwen3.8-27B` | **`52.21` tok/s** (`18.60 ms`) | **`344.50` tok/s** (`573.5 ms` / `21.13 ms`) | **`847.00` tok/s** (`670.5 ms` / `34.66 ms`) | `6,776.0` tok/s | [`p1 JSON`](./serving/results/p1_base_bf16_summary.json) |
| **`p2`: Static Multi-LoRA (`BF16` Base)** | Single adapter: `taste` (`r=128`) | `36.40` tok/s (`26.23 ms`) | `238.06` tok/s (`836.0 ms` / `29.84 ms`) | `637.74` tok/s (`855.5 ms` / `45.61 ms`) | `5,101.9` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p2`: Static Multi-LoRA (`BF16` Base)** | Mixed batch: `base` + `taste` (`r=128`) + `taiwan` (`r=16`) | — | `249.29` tok/s *(2-LoRA)* (`806.3 ms` / `28.99 ms`) | **`655.65` tok/s** (`855.9 ms` / `45.26 ms`) | `5,245.2` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — Train-to-Serve)** | **`tpu_sft` (`r=16`, TPU `v5p-16` TorchTitan adapter)** via `/v1/load_lora_adapter` (`0.16 s`) | `36.31` tok/s (`26.21 ms`) | `238.01` tok/s (`849.5 ms` / `29.64 ms`) | **`639.82` tok/s** (`855.5 ms` / `45.24 ms`) | `5,118.6` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — No Swap)** | `taste` + `taiwan` (`2` adapters in `2` HBM slots) | — | `243.55` tok/s (`838.1 ms` / `29.42 ms`) | **`654.09` tok/s** (`855.5 ms` / `45.25 ms`) | `5,232.7` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p3`: Dynamic Runtime LoRA (`BF16` Base — Live Eviction/Swap)** | **`4` concurrent adapters (`tpu_sft + taste + taiwan + jev`) -> `2` HBM slots (`max_cpu_loras=4`)** | — | `154.35` tok/s (`2,090 ms` / `28.84 ms`) | **`317.55` tok/s** (`9,833 ms` / `35.59 ms`) | `2,540.4` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p4`: `FP8`-Base QLoRA-Style Serving** | `Qwen/Qwen3.8-27B-FP8` Base (`--enable-lora` active, unadapted requests) | `49.55` tok/s (`19.19 ms`) | `319.43` tok/s (`587.7 ms` / `21.26 ms`) | **`854.93` tok/s** (`605.1 ms` / `33.36 ms`) | `6,839.4` tok/s | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |
| **`p4`: `FP8`-Base QLoRA-Style Multi-LoRA** | **Mixed `taste` (`r=128`) + `taiwan` (`r=16`) `BF16` adapters on `FP8` Base** | — | **`353.22` tok/s** *(`+41.7%` vs `BF16`)* (`523.2 ms` / **`20.73 ms`**) | **`889.78` tok/s** *(`+36.2%` vs `BF16`)* (`605.0 ms` / **`33.35 ms`**) | **`7,118.2` tok/s** | [`v3 JSON`](./serving/results/p2_p3_p4_multilora_summary.json) |

### Key Serving Findings & Runtime Patches
1. **Two Missing `vllm-torchtpu` Hooks Patched Cleanly (`0` Core Engine Edits):**
   - `TPUModelRunner.load_model` overrides `GPUModelRunner.load_model` without invoking `self.load_lora_model(...)`, causing `AttributeError: 'TPUModelRunner' object has no attribute 'lora_manager'` during warmup (`p2` unpatched baseline in [`p1_base_bf16_summary.json`](./serving/results/p1_base_bf16_summary.json)). Injecting the 3-line `load_lora_model` hook ([`run_vllm_tpu_lora_suite.py#L163-L183`](./serving/scripts/run_vllm_tpu_lora_suite.py#L163-L183)) activates Punica/Pallas batched LoRA wrappers across both full-attention and Gated-DeltaNet projections.
   - `TPUWorker` inherits `NotImplementedError` stubs from `WorkerBase` for `add_lora`, `remove_lora`, `pin_lora`, and `list_loras`; forwarding those four methods to `self.model_runner` ([`run_vllm_tpu_lora_suite.py#L188-L225`](./serving/scripts/run_vllm_tpu_lora_suite.py#L188-L225)) unlocks both `--lora-modules` startup loading and sub-second runtime `/v1/load_lora_adapter` (`0.16–0.85 s`) / `/v1/unload_lora_adapter` (`<0.01 s`).
2. **`FP8`-Base + `BF16` Multi-LoRA ("QLoRA-Style Serving") Delivers `+36.2%` Higher Throughput Than `BF16` Multi-LoRA:** Combining `Qwen3.8-27B-FP8` base weights (`~27 GB` vs `~54 GB` in HBM) with `BF16` rank-`128`/`16` adapters reaches **`889.78` tok/s** (`33.35 ms` TPOT at `c=32`), halving base-weight HBM bandwidth during decode so batched LoRA adapter overhead completely disappears relative to the `BF16` non-LoRA baseline (`847.00` tok/s).

---

## 3. Repository Layout

- [`docs/QWEN38_27B_TPU_LORA_QLORA_REPORT.md`](./docs/QWEN38_27B_TPU_LORA_QLORA_REPORT.md) — Detailed architecture, FLOP accounting, HBM breakdown, chat completion behavioral probes, and reproduction guide.
- [`finetuning/scripts/`](./finetuning/scripts/) — `lora_tpu_recipes.py` (`BF16` LoRA, `INT8` QLoRA, `NF4` QLoRA + `PEFT` exporter), `prep_sft_data.py`, `launch_ft.sh`, `ft_ctl.sh`, `ft_chain.sh`, `hbm_sampler.sh`, `parse_ft_logs.py`, `make_ft_summary.py`.
- [`finetuning/k8s/`](./finetuning/k8s/) — Indexed 4-host `TPU v5p-16` (`2x2x4`) batch and interactive harness manifests.
- [`finetuning/results/`](./finetuning/results/) — `summary.json`, per-recipe `run_stats.json`, `qlora_int8_hbm_telemetry.json`, and exported `adapter_config.json`.
- [`serving/scripts/`](./serving/scripts/) — `run_vllm_tpu_lora_suite.py` (automated `vllm-torchtpu` patcher + multi-phase OpenAI streaming benchmark + quality/chat probes) and `tpu_single_host_setup.py` (`LD_PRELOAD` single-host metadata isolator).
- [`serving/plans/`](./serving/plans/) & [`serving/k8s/`](./serving/k8s/) — Benchmark JSON plans (`p1`–`p4`) and `TPU v7x` Kubernetes Job manifest.
- [`serving/results/`](./serving/results/) — Complete `p1`–`p4` latency/throughput JSONs and greedy generation transcripts.
