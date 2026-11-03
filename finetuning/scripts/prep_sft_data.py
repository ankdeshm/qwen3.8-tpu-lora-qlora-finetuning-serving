#!/usr/bin/env python3
"""Prepare a single-turn SFT dataset (yahma/alpaca-cleaned) as chat `messages` JSONL for torchtitan.

Output rows: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
which torchtitan's ChatProcessor renders with the model's own chat template (Qwen3.8-27B
chat_template.jinja) and masks the prompt tokens out of the loss.

Also renders + tokenizes a few samples with torchtitan's HuggingFaceTokenizer to validate the template.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yahma/alpaca-cleaned")
    ap.add_argument("--file", default="alpaca_data_cleaned.json")
    ap.add_argument("--out", default="/models/lora-ft/data/alpaca_cleaned_chat.jsonl")
    ap.add_argument("--tokenizer", default="/models/models/Qwen/Qwen3.8-27B")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    src = hf_hub_download(args.repo, args.file, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    rows = json.loads(Path(src).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = Path("/tmp") / out.name
    with open(tmp, "w") as f:
        for r in rows:
            instr = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            outp = (r.get("output") or "").strip()
            if not instr or not outp:
                continue
            user = instr if not inp else f"{instr}\n\n{inp}"
            f.write(json.dumps({"messages": [{"role": "user", "content": user},
                                             {"role": "assistant", "content": outp}]}, ensure_ascii=False) + "\n")
            n += 1
    out.write_bytes(tmp.read_bytes())
    print(f"wrote {n} rows -> {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Validate the chat template + tokenization path exactly as ChatProcessor uses it.
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tok = HuggingFaceTokenizer(tokenizer_path=args.tokenizer)
    print("bos_id", tok.bos_id, "eos_id", tok.eos_id, "eos_token", tok.eos_token)
    lens = []
    with open(out) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            msgs = json.loads(line)["messages"]
            full = tok.apply_chat_template(msgs).rstrip("\n")
            ids = tok.encode(full, add_bos=True, add_eos=False)
            lens.append(len(ids))
            if i == 0:
                prompt = tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
                print("---- rendered sample 0 ----\n" + full + "\n---- prompt part ----\n" + prompt)
                print("prompt tokens", len(tok.encode(prompt, add_bos=True, add_eos=False)), "full tokens", len(ids))
    lens.sort()
    print(f"token length over first {len(lens)} samples: mean={sum(lens) / len(lens):.0f} "
          f"p50={lens[len(lens) // 2]} p99={lens[int(0.99 * (len(lens) - 1))]} max={lens[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
