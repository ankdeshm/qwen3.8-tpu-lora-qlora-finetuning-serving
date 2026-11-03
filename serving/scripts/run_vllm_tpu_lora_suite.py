#!/usr/bin/env python3
"""vLLM-TorchTPU LoRA serving tests on ONE TPU device (the TPU LoRA path supports TP=1, dense models, no spec decode).

Phases (JSON plan): base model; static multi-LoRA (adapters registered at startup, requests per adapter and mixed
across adapters); dynamic LoRA (adapters loaded/unloaded at runtime through /v1/load_lora_adapter with fewer device
slots than adapters, forcing swaps); and LoRA on an FP8 base checkpoint ("QLoRA-style" serving).

The benchmark client is self-contained: fixed-length random token-id prompts, ignore_eos, streaming completions;
reports output tok/s, TTFT and TPOT percentiles. A quality probe compares base vs adapter completions (greedy).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/scripts")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}][lora-suite] {msg}", flush=True)


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def stream_one(port: int, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                    n += 1
                    if first is None:
                        first = time.time()
        t1 = time.time()
        return {"ok": True, "ttft": (first or t1) - t0, "e2e": t1 - t0, "chunks": n,
                "tpot": ((t1 - first) / max(n - 1, 1)) if first else None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_bench(port: int, models: list[str], inp: int, out: int, conc: int, vocab: int, seed: int = 0) -> dict:
    rng = random.Random(seed + inp * 7 + conc)
    n_req = max(4 * conc, 8)
    jobs = [(models[i % len(models)], [rng.randrange(1000, vocab - 1000) for _ in range(inp)]) for i in range(n_req)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda j: stream_one(port, j[0], j[1], out), jobs))
    wall = time.time() - t0
    ok = [r for r in res if r.get("ok")]
    toks = sum(r["chunks"] for r in ok)
    ttft = [r["ttft"] * 1000 for r in ok]
    tpot = [r["tpot"] * 1000 for r in ok if r.get("tpot")]
    return {
        "models": models, "input_len": inp, "output_len": out, "concurrency": conc, "requests": n_req,
        "succeeded": len(ok), "failed": len(res) - len(ok), "wall_s": round(wall, 2),
        "output_tok_s": round(toks / wall, 2) if wall else 0.0,
        "ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p99": pct(ttft, 0.99),
        "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p99": pct(tpot, 0.99),
        "errors": list({r.get("error", "") for r in res if not r.get("ok")})[:3],
    }


def quality_probe(port: int, models: list[str]) -> dict:
    prompt = "Write two sentences about the city of Taipei and its food."
    outs = {}
    for m in models:
        code, txt = http_json(f"http://127.0.0.1:{port}/v1/completions",
                              {"model": m, "prompt": prompt, "max_tokens": 64, "temperature": 0.0})
        try:
            outs[m] = json.loads(txt)["choices"][0]["text"][:300] if code == 200 else f"HTTP {code}: {txt[:200]}"
        except Exception:
            outs[m] = txt[:200]
    return outs


def chat_probe(port: int, models: list[str], prompts: list[str], max_tokens: int = 160) -> dict:
    """Greedy chat completions (model's own chat template) for each model x prompt; records text + token counts."""
    outs: dict[str, list[dict]] = {}
    for m in models:
        outs[m] = []
        for p in prompts:
            t0 = time.time()
            code, txt = http_json(f"http://127.0.0.1:{port}/v1/chat/completions",
                                  {"model": m, "messages": [{"role": "user", "content": p}],
                                   "max_tokens": max_tokens, "temperature": 0.0})
            rec = {"prompt": p, "http": code, "s": round(time.time() - t0, 2)}
            try:
                j = json.loads(txt)
                rec["text"] = j["choices"][0]["message"].get("content") or ""
                rec["reasoning"] = (j["choices"][0]["message"].get("reasoning_content") or "")[:400]
                rec["finish_reason"] = j["choices"][0].get("finish_reason")
                rec["completion_tokens"] = j.get("usage", {}).get("completion_tokens")
                rec["text"] = rec["text"][:600]
            except Exception:
                rec["text"] = txt[:300]
            outs[m].append(rec)
    return outs


def stage(gcs: str, dst: Path) -> None:
    if (dst / ".staged").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", "-r", "--no-user-output-enabled", f"{gcs.rstrip('/')}/*", str(dst)], check=True)
    (dst / ".staged").write_text("ok")


def fetch_adapter(a: dict, root: Path) -> Path:
    if a.get("path"):  # local / GCS-FUSE adapter dir (e.g. the adapter fine-tuned with torchtitan on v5p)
        dst = root / a["name"] / "_local"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            subprocess.run(["cp", str(Path(a["path"]) / f), str(dst / f)], check=True)
        return dst
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(a["repo"], allow_patterns=a["allow"], local_dir=str(root / a["name"] / "_dl"),
                               token=os.environ.get("HF_TOKEN")))
    return d / a["subdir"] if a.get("subdir") else d


TPU_RUNNER = Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/runner/tpu_runner.py")
LORA_PATCH_MARK = "[lora-suite patch]"


def patch_tpu_runner_lora() -> str:
    """vllm-torchtpu (image vllm-torchtpu-bench:v0.32.0): TPUModelRunner.load_model overrides
    GPUModelRunner.load_model but never calls LoRAModelRunnerMixin.load_lora_model, so with
    --enable-lora the engine dies in profile_run (AttributeError: no attribute 'lora_manager').
    Mirror gpu_model_runner.load_model: wrap the model with the LoRA manager right after loading.
    Patched in the (ephemeral) container source before any `vllm serve` process starts."""
    if not TPU_RUNNER.exists():
        return "tpu_runner.py not found"
    src = TPU_RUNNER.read_text()
    if LORA_PATCH_MARK in src:
        return "already patched"
    anchor = "        validate_online_fp8(model, self.vllm_config.quant_config)\n        self.model = model\n"
    if anchor not in src:
        return "anchor not found (not patched)"
    inject = anchor + (
        "        if self.lora_config is not None and getattr(self, 'lora_manager', None) is None:\n"
        f"            # {LORA_PATCH_MARK} wire LoRA like GPUModelRunner.load_model does\n"
        "            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)\n")
    TPU_RUNNER.write_text(src.replace(anchor, inject, 1))
    return "patched: TPUModelRunner.load_model now calls load_lora_model when --enable-lora"


TPU_WORKER = Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/worker/tpu_worker.py")
WORKER_PATCH = f'''

# {LORA_PATCH_MARK} TPUWorker derives from WorkerBase, whose add_lora/remove_lora/pin_lora/list_loras
# raise NotImplementedError; forward them to the model runner exactly like vLLM's GPUWorker does.
def _lora_suite_add_lora(self, lora_request):
    return self.model_runner.add_lora(lora_request)


def _lora_suite_remove_lora(self, lora_id):
    return self.model_runner.remove_lora(lora_id)


def _lora_suite_pin_lora(self, lora_id):
    return self.model_runner.pin_lora(lora_id)


def _lora_suite_list_loras(self):
    return self.model_runner.list_loras()


for _lora_suite_name, _lora_suite_fn in (("add_lora", _lora_suite_add_lora), ("remove_lora", _lora_suite_remove_lora),
                                         ("pin_lora", _lora_suite_pin_lora), ("list_loras", _lora_suite_list_loras)):
    if _lora_suite_name not in TPUWorker.__dict__:
        setattr(TPUWorker, _lora_suite_name, _lora_suite_fn)
'''


def patch_tpu_worker_lora() -> str:
    """vllm-torchtpu (vllm-torchtpu-bench:v0.32.0): TPUWorker has no LoRA RPCs, so adding an adapter
    (static --lora-modules at startup or /v1/load_lora_adapter) fails with NotImplementedError."""
    if not TPU_WORKER.exists():
        return "tpu_worker.py not found"
    src = TPU_WORKER.read_text()
    if "_lora_suite_add_lora" in src:
        return "already patched"
    if "class TPUWorker" not in src:
        return "TPUWorker not found (not patched)"
    TPU_WORKER.write_text(src + WORKER_PATCH)
    return "patched: TPUWorker.add_lora/remove_lora/pin_lora/list_loras forward to the model runner"


def main() -> int:
    plan = json.loads(Path(sys.argv[1]).read_text())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(plan["out_root"]) / f"{plan['tag']}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    port = int(plan.get("port", 8000))
    patch_state = patch_tpu_runner_lora()
    log(f"runtime patch: {patch_state}")
    from tpu_single_host_setup import start_single_host_metadata_env
    child_env = start_single_host_metadata_env("v7x")
    child_env.update({"VLLM_PLUGINS": "torchtpu", "VLLM_TARGET_DEVICE": "tpu", "PJRT_DEVICE": "TPU",
                      "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1",
                      "VLLM_ENGINE_READY_TIMEOUT_S": "5400", "HF_HOME": "/dev/shm/hf_cache"})
    adapters = {}
    specs = {a["name"]: a for a in plan["adapters"]}

    def ensure_adapter(name: str) -> None:
        if name in adapters or name not in specs:
            return
        a = specs[name]
        if a.get("path"):
            deadline = time.time() + float(a.get("wait_s", 0))
            while not (Path(a["path"]) / "adapter_model.safetensors").exists() and time.time() < deadline:
                time.sleep(20)
        try:
            adapters[name] = str(fetch_adapter(a, Path("/dev/shm/adapters")))
            log(f"adapter {name} -> {adapters[name]}")
        except Exception as e:
            log(f"adapter {name} fetch failed: {e}")

    for a in plan["adapters"]:
        if not a.get("path"):
            ensure_adapter(a["name"])
    worker_patch = patch_tpu_worker_lora()
    log(f"runtime patch: {worker_patch}")
    results = {"plan": plan, "adapters": adapters, "phases": [],
               "runtime_patches": {"vllm_torchtpu.runner.tpu_runner.TPUModelRunner.load_model": patch_state,
                                   "vllm_torchtpu.worker.tpu_worker.TPUWorker LoRA RPCs": worker_patch}}
    staged: dict[str, Path] = {}
    for ph in plan["phases"]:
        rec = {"name": ph["name"], "desc": ph["desc"], "status": "PENDING", "bench": []}
        results["phases"].append(rec)
        try:
            for n in (ph.get("lora") or {}).get("adapters", []):
                ensure_adapter(n)
            if ph.get("model_path"):
                model_dir = ph["model_path"]  # e.g. the GCS FUSE mount; read directly by the single TP=1 worker
            else:
                if ph["model_gcs"] not in staged:
                    for g, p in list(staged.items()):
                        subprocess.run(["rm", "-rf", str(p)], check=False)
                        staged.pop(g)
                    staged[ph["model_gcs"]] = Path("/dev/shm/models") / ph["model_gcs"].rstrip("/").split("/")[-1]
                    t_stage = time.time()
                    stage(ph["model_gcs"], staged[ph["model_gcs"]])
                    rec["stage_s"] = round(time.time() - t_stage, 1)
                model_dir = str(staged[ph["model_gcs"]])
            lora = ph.get("lora")
            cmd = ["vllm", "serve", model_dir, f"--served-model-name={ph['served']}", "--host=127.0.0.1",
                   f"--port={port}", "--tensor-parallel-size=1", f"--max-model-len={ph.get('max_model_len', 8192)}",
                   f"--max-num-seqs={ph.get('max_num_seqs', 64)}", f"--gpu-memory-utilization={ph.get('mem_util', 0.9)}",
                   "--trust-remote-code", *ph.get("extra_args", [])]
            env = dict(child_env)
            if lora:
                cmd += ["--enable-lora", f"--max-loras={lora['max_loras']}", f"--max-lora-rank={lora['max_rank']}",
                        f"--max-cpu-loras={lora.get('max_cpu_loras', max(lora['max_loras'], len(lora['adapters'])))}"]
                if lora["mode"] == "static":
                    cmd += ["--lora-modules", *[f"{n}={adapters[n]}" for n in lora["adapters"] if n in adapters]]
                else:
                    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
            rec["command"] = cmd
            log(f"[{ph['name']}] {' '.join(cmd)}")
            slog = out / f"server_{ph['name']}.log"
            with open(slog, "w") as lf:
                proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            t0 = time.time()
            ready = False
            while time.time() - t0 < float(ph.get("ready_timeout_s", 5400)):
                if proc.poll() is not None:
                    break
                if http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)[0] == 200:
                    ready = True
                    break
                time.sleep(10)
            rec["boot_s"] = round(time.time() - t0, 1)
            if not ready:
                rec["status"] = "SERVER_FAILED"
                rec["server_log_tail"] = slog.read_text(errors="replace")[-4000:]
                log(f"[{ph['name']}] server failed after {rec['boot_s']}s")
            else:
                vocab = int(ph.get("vocab", 151000))
                if lora and lora["mode"] == "dynamic":
                    rec["dynamic_ops"] = []
                    for n in lora["adapters"]:
                        if n not in adapters:
                            continue
                        t1 = time.time()
                        code, txt = http_json(f"http://127.0.0.1:{port}/v1/load_lora_adapter",
                                              {"lora_name": n, "lora_path": adapters[n]})
                        rec["dynamic_ops"].append({"op": "load", "adapter": n, "http": code, "s": round(time.time() - t1, 2), "msg": txt[:200]})
                    code, txt = http_json(f"http://127.0.0.1:{port}/v1/models")
                    rec["models_after_load"] = txt[:1000]
                live = [n for n in (lora or {}).get("adapters", []) if n in adapters]
                if lora and lora["mode"] == "dynamic":
                    live = [o["adapter"] for o in rec["dynamic_ops"] if o["op"] == "load" and o["http"] == 200]
                rec["live_adapters"] = live
                rec["quality_probe"] = quality_probe(port, [ph["served"]] + live)
                if ph.get("chat_probe"):
                    rec["chat_probe"] = chat_probe(port, [ph["served"]] + live, ph["chat_probe"],
                                                   int(ph.get("chat_probe_max_tokens", 160)))
                    (out / "summary.json").write_text(json.dumps(results, indent=2))
                for b in ph["bench"]:
                    models = [ph["served"] if m == "base" else m for m in b["models"] if m == "base" or m in live]
                    if not models:
                        rec["bench"].append({"models": b["models"], "skipped": "no requested adapter is live"})
                        continue
                    for c in b["conc"]:
                        r = run_bench(port, models, b["in"], b["out"], c, vocab)
                        rec["bench"].append(r)
                        log(f"[{ph['name']}] {models} in={b['in']} out={b['out']} c={c} -> {r['output_tok_s']} tok/s "
                            f"ttft_p50={r['ttft_ms_p50']} tpot_p50={r['tpot_ms_p50']} failed={r['failed']}")
                        (out / "summary.json").write_text(json.dumps(results, indent=2))
                if lora and lora["mode"] == "dynamic" and lora.get("unload"):
                    for n in lora["unload"]:
                        t1 = time.time()
                        code, txt = http_json(f"http://127.0.0.1:{port}/v1/unload_lora_adapter", {"lora_name": n})
                        rec["dynamic_ops"].append({"op": "unload", "adapter": n, "http": code, "s": round(time.time() - t1, 2), "msg": txt[:200]})
                rec["status"] = "DONE"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
            time.sleep(20)
        except Exception as e:
            rec["status"] = f"ERROR: {type(e).__name__}: {e}"[:400]
            log(f"[{ph['name']}] {rec['status']}")
        (out / "summary.json").write_text(json.dumps(results, indent=2))
    log(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
