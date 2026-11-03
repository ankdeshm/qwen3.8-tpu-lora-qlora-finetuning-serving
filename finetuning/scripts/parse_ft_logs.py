#!/usr/bin/env python3
"""Summarize torchtitan LoRA / QLoRA fine-tuning logs (rank-0 / node-0 log) into JSON + Markdown.

FLOP conventions (per trained token, Qwen3.8-27B, seq 4096, causal):
  N_mm   = matmul params touched per token (decoder linears + lm_head; embedding lookup excluded)
  attn   = 16 full-attention layers: fwd 2*2*S*(n_heads*head_dim)*0.5 per layer (causal half)
  LoRA "model FLOPs" (MFU convention used here)
         = fwd 2*N_mm + bwd activation-grad 2*N_mm (frozen weights get no weight-grad)
           + 3*attn_fwd + 6*N_lora
  LoRA "hardware FLOPs" (HFU) = model FLOPs + recompute of the forward under full activation
           checkpointing (+2*N_mm + attn_fwd)
  Full fine-tune reference = 6*N_mm + 3*attn_fwd (what torchtitan's logged `mfu` assumes).
GDN chunked-delta-rule FLOPs (~1-2% of the total) are not counted in any convention.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[0-9.]+).*?grad_norm:\s*(?P<gn>[0-9.]+).*?"
                     r"memory:\s*(?P<mem>[0-9.]+)GiB\((?P<mempct>[0-9.]+)%\).*?tps:\s*(?P<tps>[0-9,]+).*?"
                     r"tflops:\s*(?P<tflops>[0-9.]+).*?mfu:\s*(?P<mfu>[0-9.]+)%.*?time_end_to_end:\s*(?P<t>[0-9.]+)s")

PEAK_BF16 = {"v5p": 459e12, "v7x_per_core": 1153.5e12}

# Qwen3.8-27B (text decoder) dimensions
D, L, L_ATTN, V, FFN = 5120, 64, 16, 248320, 17408
N_HEADS, HEAD_DIM, N_KV = 24, 256, 4
GDN_K, GDN_V = 16 * 128, 48 * 128


def qwen38_27b_counts(seq: int, lora_rank: int = 16) -> dict:
    mlp = 3 * D * FFN * L
    attn_proj = (D * N_HEADS * HEAD_DIM * 2 + 2 * D * N_KV * HEAD_DIM + N_HEADS * HEAD_DIM * D) * L_ATTN
    gdn = (2 * D * GDN_K + 2 * D * GDN_V + 2 * D * 48 + GDN_V * D) * (L - L_ATTN)
    lm_head = D * V
    n_mm = mlp + attn_proj + gdn + lm_head
    r = lora_rank
    lora = (r * (D + N_HEADS * HEAD_DIM * 2) + 2 * r * (D + N_KV * HEAD_DIM) + r * (N_HEADS * HEAD_DIM + D)) * L_ATTN \
        + 3 * r * (D + FFN) * L
    attn_fwd = 2 * 2 * seq * (N_HEADS * HEAD_DIM) * 0.5 * L_ATTN
    model = 4 * n_mm + 3 * attn_fwd + 6 * lora
    return {"n_matmul_params": n_mm, "n_lora_params_formula": lora, "attn_fwd_flops_per_tok": attn_fwd,
            "lora_model_flops_per_tok": model, "lora_hw_flops_per_tok": model + 2 * n_mm + attn_fwd,
            "full_ft_flops_per_tok": 6 * n_mm + 3 * attn_fwd}


def parse(log_path: Path, steady_from: int, chip: str = "v5p", seq: int = 4096, lora_rank: int = 16) -> dict:
    steps: dict[int, dict] = {}
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = ansi_re.sub("", raw_line)
        if "[titan]" not in line:
            continue
        m = STEP_RE.search(line)
        if not m:
            continue
        s = int(m["step"])
        steps.setdefault(s, {"step": s, "loss": float(m["loss"]), "grad_norm": float(m["gn"]),
                             "mem_gib": float(m["mem"]), "mem_pct": float(m["mempct"]),
                             "tps": float(m["tps"].replace(",", "")), "logged_tflops": float(m["tflops"]),
                             "logged_mfu_pct": float(m["mfu"]), "step_time_s": float(m["t"])})
    rows = [steps[k] for k in sorted(steps)]
    c = qwen38_27b_counts(seq, lora_rank)
    peak = PEAK_BF16["v5p"] if chip == "v5p" else PEAK_BF16["v7x_per_core"]
    steady = [r for r in rows if r["step"] >= steady_from]
    out = {"num_steps_logged": len(rows), "steady_from_step": steady_from, "flop_model": c, "steps": rows}
    if steady:
        tps = sum(r["tps"] for r in steady) / len(steady)
        st = sum(r["step_time_s"] for r in steady) / len(steady)
        out.update({
            "steady_steps": len(steady),
            "steady_tok_s_per_chip": round(tps, 1),
            "steady_step_time_s": round(st, 3),
            "peak_mem_gib_per_chip_max": max(r["mem_gib"] for r in rows),
            "steady_mem_gib_per_chip": steady[-1]["mem_gib"],
            "mfu_lora_model_flops_pct": round(100 * tps * c["lora_model_flops_per_tok"] / peak, 2),
            "hfu_with_recompute_pct": round(100 * tps * c["lora_hw_flops_per_tok"] / peak, 2),
            "mfu_if_counted_as_full_ft_6N_pct": round(100 * tps * c["full_ft_flops_per_tok"] / peak, 2),
            "torchtitan_logged_mfu_pct_mean": round(sum(r["logged_mfu_pct"] for r in steady) / len(steady), 2),
            "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
            "loss_mean_last10": round(sum(r["loss"] for r in rows[-10:]) / min(10, len(rows)), 4),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="name=path/to/node0.log")
    ap.add_argument("--steady-from", type=int, default=11)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = {}
    for spec in args.logs:
        name, path = spec.split("=", 1)
        res[name] = parse(Path(path), args.steady_from)
        r = res[name]
        print(name, {k: r.get(k) for k in ("num_steps_logged", "steady_tok_s_per_chip", "steady_step_time_s",
                                            "peak_mem_gib_per_chip_max", "mfu_lora_model_flops_pct",
                                            "hfu_with_recompute_pct", "loss_first", "loss_last")})
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
