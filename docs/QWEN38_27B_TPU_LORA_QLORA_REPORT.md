# Technical Report: `Qwen3.8-27B` LoRA & QLoRA Fine-Tuning (`v5p-16`) + Multi-LoRA Serving (`v7x`)

## 1. Architecture & Hybrid Target Mapping

`Qwen/Qwen3.8-27B` uses the `Qwen3_5ForConditionalGeneration` hybrid text decoder comprising **`64` layers** (`d_model=5120`, `ffn=17408`, `vocab=248320`, `26,975,690,240` parameters):
- **`48` Gated-DeltaNet (`linear_attn`) layers** + **`64` SwiGLU (`gate_proj, up_proj, down_proj`) MLP blocks**
- **`16` Gated Full-Attention (`self_attn`) layers** (`24` query heads, `4` KV heads, `head_dim=256`)

In TorchTitan's TPU lane (`torchtitan.experiments.tpu.qwen3_5`), applying `LoRAConverter(rank=16, alpha=32)` across `wq, wk, wv, wo, w1, w2, w3` attaches **`256` low-rank adapter pairs (`512` matrices, `79,691,776` trainable parameters = `0.2954%`)**. At the end of training, [`export_peft_adapter` in `lora_tpu_recipes.py`](../finetuning/scripts/lora_tpu_recipes.py#L465-L505) gathers the `DTensor` shards in `BF16` and maps TorchTitan module names onto Hugging Face `PEFT` paths (`base_model.model.model.language_model.layers.<L>.{self_attn,mlp}.*`), producing a `159.4 MB` `adapter_model.safetensors` ready for immediate serving in `vLLM`.

---

## 2. FLOP & HBM Accounting on `TPU v5p-16` (`2x2x4`, `16` Chips)

Per trained token at `seq_len=4096`:
- Matmul parameters touched (`N_mm`, decoder linears + `lm_head`, excluding embedding lookup): **`26.906B`**
- Causal full-attention forward FLOPs across `16` full-attention layers (`attn_fwd`): **`0.403 GFLOP/tok`**
- **LoRA Model FLOPs (`4 * N_mm + 3 * attn_fwd + 6 * N_lora`):** **`109.31 GFLOP/tok`** *(frozen base weights require forward `2N` + activation-gradient backward `2N`, skipping the `2N` weight-gradient GEMM)*
- **Hardware FLOPs under `FullAC` (`+2 * N_mm + attn_fwd` forward recomputation):** **`163.52 GFLOP/tok`**

### HBM Footprint Comparison (`SimpleFSDP` ZeRO-3 Across `16` Chips)

| Stage / Metric | `lora_bf16` (`16`-bit Base) | `qlora_int8` (`8`-bit Base) | `qlora_nf4` (`4.125`-bit Base) |
|---|---:|---:|---:|
| Post-HF-Load Allocated HBM / Chip | `4.549 GiB` | `4.549 GiB` | `4.549 GiB` |
| Post-Quantization Resting HBM / Chip | `4.549 GiB` *(unquantized)* | **`3.143 GiB` (`-30.9%`)** | **`2.474 GiB` (`-45.6%`)** |
| Quantization Time (`592` Linear Modules, `24.35B` Params) | `0.0 s` | `3.8 s` (`rel_err=0.0104`) | `4.7 s` (`rel_err=0.0927`) |
| Steady Step Time (`131,072` tok/step, steps `11..150`) | `9.772 s` | **`9.357 s` (`-4.2%`)** | `13.226 s` |
| Steady Throughput / Chip (`16`-Chip Aggregate) | `838.9` (`13,422` tok/s) | **`876.0` (`14,016` tok/s)** | `619.6` (`9,914` tok/s) |
| Total `150`-Step Training Wall Time | `1,574.8 s` | **`1,427.4 s` (`-147.4 s`)** | `2,007.1 s` |
| Peak Device Allocated HBM / Chip | `32.415 GiB` | **`30.906 GiB`** | `42.767 GiB` |
| `tpu-info` Runtime Sampled HBM (`min / mean / max`) | — | **`0.0 / 23.33 / 41.98 GiB`** | — |
| Step 1 Loss -> Last-10 Mean Loss (`Step 150`) | `1.1522 -> 0.8498` (`0.8416`) | `1.1555 -> 0.8500` (`0.8417`) | `1.1221 -> 0.8543` (`0.8462`) |

---

## 3. End-to-End Train-to-Serve Verification on `TPU v7x`

In Phase `p3_dynamic_lora_bf16_incl_tpu_trained` ([`p2_p3_p4_multilora_summary.json`](../serving/results/p2_p3_p4_multilora_summary.json)), `vllm-torchtpu` boots `Qwen/Qwen3.8-27B` with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `--max-loras=2`, `--max-cpu-loras=4`, and `--max-lora-rank=128`, then executes live HTTP control-plane operations:

1. `POST /v1/load_lora_adapter` `tpu_sft` (`r=16`, trained on `v5p-16` in Section 2) -> **`HTTP 200` in `0.16 s`**
2. `POST /v1/load_lora_adapter` `taste` (`r=128`, `Moeblack/Qwen3.8-27B-chinese-taste-lora`) -> **`HTTP 200` in `0.85 s`**
3. `POST /v1/load_lora_adapter` `taiwan` (`r=16`, `noisebridge/qwen3.8-27b-taiwan-pilled`) -> **`HTTP 200` in `0.22 s`**
4. `POST /v1/load_lora_adapter` `jev` (`r=32`, `autotrust/JEV-27B-VL`, targeting `self_attn`, `linear_attn` GDN projections, `mlp`, and `lm_head`) -> **`HTTP 200` in `0.58 s`**
5. Greedy `/v1/chat/completions` behavioral verification confirms `tpu_sft` cleanly suppresses base `<think>...</think>` chain-of-thought tags in favor of direct Alpaca-style instruction responses while maintaining **`639.82` tok/s** (`c=32`, `0` errors).
6. `POST /v1/unload_lora_adapter` `taste` -> **`HTTP 200` in `<0.01 s`**.

---

## 4. Reproducing Fine-Tuning & Serving

### 4.1 Fine-Tuning (`TPU v5p-16`)
```bash
kubectl create configmap lora-ft-code \
  --from-file=finetuning/scripts/lora_tpu_recipes.py \
  --from-file=finetuning/scripts/launch_ft.sh \
  --from-file=finetuning/scripts/prep_sft_data.py
kubectl apply -f finetuning/k8s/lora-ft-qwen38-27b-v5p16.yaml
```

### 4.2 Multi-LoRA & `FP8`-Base Serving (`TPU v7x`)
```bash
kubectl create configmap lora-tpu-scripts \
  --from-file=serving/scripts/run_vllm_tpu_lora_suite.py \
  --from-file=serving/scripts/tpu_single_host_setup.py \
  --from-file=plan.json=serving/plans/plan_base_p1.json \
  --from-file=plan_v3.json=serving/plans/plan_multilora_p2_p3_p4.json
kubectl apply -f serving/k8s/lora-serve-qwen38-27b-v7x.yaml
```
