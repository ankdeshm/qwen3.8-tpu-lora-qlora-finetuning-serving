#!/usr/bin/env python3
"""Build the customer-ready fine-tuning results folder (summary.json + SUMMARY.md) from node-0 logs + run_stats.json.

usage: make_ft_summary.py --out <dir> name=<node0.log>:<run_stats.json> [...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import parse_ft_logs as P

DESC = {
    "lora_bf16": "LoRA, frozen BF16 base (2 B/param)",
    "qlora_int8": "QLoRA-style, frozen INT8 weight-only base (per-output-channel absmax, ~1 B/param)",
    "qlora_nf4": "QLoRA-format, frozen NF4 base (QLoRA codebook, block 64, 8-bit double-quant absmax, 4.125 bit/param)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steady-from", type=int, default=11)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.runs:
        name, paths = spec.split("=", 1)
        log, stats = paths.split(":", 1)
        r = P.parse(Path(log), a.steady_from)
        st = json.loads(Path(stats).read_text()) if Path(stats).exists() else {}
        r["run_stats"] = st
        runs[name] = r
    summary = {
        "what": "Qwen3.8-27B LoRA vs QLoRA-style fine-tuning with torchtitan on TorchTPU 0.2.0, TPU v5p-16 (2x2x4, 16 chips)",
        "hardware": {"slice": "TPU v5p 2x2x4 (4 hosts x 4 chips)", "nodepool": "tpu-v5p-16-pool",
                     "peak_bf16_tflops_per_chip": 459, "hbm_gib_per_chip": 95},
        "software": {"image": "us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/torchtitan-flint:v0.2.0",
                     "torch": "2.15.0+cpu", "torch_tpu": "0.2.0", "torchtitan": "/torchtitan-native (TPU lane)"},
        "model": {"name": "Qwen/Qwen3.8-27B", "arch": "Qwen3_5ForConditionalGeneration text decoder: 64 layers "
                  "(48 Gated-DeltaNet + 16 gated full attention), d=5120, ffn=17408, vocab=248320",
                  "torchtitan_flavor": "torchtitan.experiments.tpu.qwen3_5 '27B' (text-only, vision tower dropped)",
                  "checkpoint": "gs://YOUR_GCS_BUCKET/models/Qwen/Qwen3.8-27B (HF safetensors, BF16)"},
        "common_config": {
            "lora": "rank 16, alpha 32 (scale 2.0), dropout 0, targets q_proj,k_proj,v_proj,o_proj (16 full-attn layers) + "
                    "gate_proj,up_proj,down_proj (64 layers); A ~ kaiming-uniform, B = 0; FP32 master + AdamW state, BF16 compute",
            "data": "yahma/alpaca-cleaned (51,760 single-turn rows) rendered with the Qwen3.8 chat template; prompt tokens "
                    "masked from the loss (torchtitan ChatProcessor); concat-then-split packing into 4096-token rows "
                    "with document masking; shuffle seed 42",
            "batch": "2 x 4096 tokens per chip, 16 chips -> 131,072 tokens/step, no grad accumulation",
            "optimizer": "AdamW lr 2e-4, betas (0.9, 0.95), wd 0, warmup 10, cosine decay over 90% of steps to 0.1x, clip 1.0",
            "parallelism": "SimpleFSDP (ZeRO-3, reshard after forward) over 16 chips; full activation checkpointing; "
                           "torch_tpu DEFER_AND_FUSE eager mode; compiled grad-norm",
            "steps": 150, "seed": 42,
            "flop_convention": "MFU uses LoRA model FLOPs = 4*N_mm (fwd + activation-grad bwd; frozen W get no weight-grad) "
                               "+ 3*attn_fwd + 6*N_lora; HFU adds the FullAC recompute (+2*N_mm + attn_fwd). torchtitan's "
                               "logged 'mfu' counts 6*N (full fine-tune) and is reported separately. GDN chunked-scan FLOPs "
                               "(~1-2%) are not counted.",
        },
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "steps"} for k, v in runs.items()},
        "loss_curves": {k: [(s["step"], s["loss"]) for s in v["steps"]] for k, v in runs.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Qwen3.8-27B LoRA vs QLoRA-style fine-tuning on TPU v5p-16 (torchtitan + TorchTPU 0.2.0)", ""]
    lines += ["| run | frozen base | trainable / total params | step time (s) | tok/s/chip | peak HBM/chip (GiB) | MFU (LoRA FLOPs) | HFU (+recompute) | torchtitan logged mfu (6N) | loss step1 -> last10 mean |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in runs.items():
        st = r.get("run_stats", {})
        tr, tot = st.get("trainable_params"), st.get("total_params")
        lines.append(f"| {k} | {DESC.get(k, k)} | {tr:,} / {tot:,} | {r.get('steady_step_time_s')} | {r.get('steady_tok_s_per_chip')} | "
                     f"{r.get('peak_mem_gib_per_chip_max')} | {r.get('mfu_lora_model_flops_pct')}% | {r.get('hfu_with_recompute_pct')}% | "
                     f"{r.get('torchtitan_logged_mfu_pct_mean')}% | {r.get('loss_first')} -> {r.get('loss_mean_last10')} |"
                     if tr and tot else f"| {k} | {DESC.get(k, k)} | n/a | {r.get('steady_step_time_s')} | {r.get('steady_tok_s_per_chip')} | "
                     f"{r.get('peak_mem_gib_per_chip_max')} | {r.get('mfu_lora_model_flops_pct')}% | {r.get('hfu_with_recompute_pct')}% | "
                     f"{r.get('torchtitan_logged_mfu_pct_mean')}% | {r.get('loss_first')} -> {r.get('loss_mean_last10')} |")
    lines += ["", "Steady state = steps >= %d. Per-chip HBM is torchtitan's per-step peak (reserved)." % a.steady_from, ""]
    (out / "SUMMARY_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
