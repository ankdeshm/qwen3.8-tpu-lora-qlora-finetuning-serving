"""LoRA and QLoRA-style fine-tuning recipes for torchtitan on TorchTPU (TPU v5p / v7x).

Model: Qwen3.8-27B (dense hybrid: 48 Gated-DeltaNet + 16 gated full-attention layers, HF arch
``Qwen3_5ForConditionalGeneration``) through the text-only TPU lane
``torchtitan.experiments.tpu.qwen3_5`` (flavor "27B" has exactly the Qwen3.8-27B dims). The base
weights are read from the HF safetensors checkpoint (``model.language_model.*`` keys).

Recipes (``--module=lora_tpu_recipes --config=<name>``):

* ``qwen38_27b_lora_bf16``  -- LoRA: frozen BF16 base weights + trainable LoRA adapters
  (FP32 master weights / AdamW state, BF16 compute, FP32 gradient reduce).
* ``qwen38_27b_qlora_int8`` -- QLoRA-style: the same adapters on a frozen INT8 weight-only base.
  After the HF checkpoint is loaded, every frozen ``nn.Linear`` inside the decoder layers is
  quantized in place to int8 with one symmetric absmax scale per output channel (scales kept in
  BF16). The int8 shards stay FSDP-sharded (1 byte/param at rest), are all-gathered as int8 and
  dequantized inside the matmul (``y = (x @ q^T) * s``). Embedding, lm_head, norms, depthwise convs
  and the GDN gates stay BF16 (as in QLoRA, which also keeps them in 16 bit).
* ``qwen38_27b_qlora_nf4``  -- QLoRA storage format on TPU: NF4 codebook (QLoRA's 4-bit NormalFloat),
  block 64 along the input dim, 8-bit double-quantized block absmax (4.125 bit/param), one uint8
  tensor per Linear (FSDP-sharded, all-gathered as uint8), decoded with a select tree + BF16 matmul.
  NOTE: original QLoRA = NF4 + double quantization via bitsandbytes CUDA kernels (paged optimizers
  too). The TPU variants here use plain PyTorch/XLA ops; they reproduce the *storage format and math*
  (frozen quantized base, BF16 compute, BF16/FP32 LoRA), not the bitsandbytes kernels.

LoRA math is torchtitan's ``LoRAConverter`` (y = W x + alpha/r * B A x, A ~ kaiming, B = 0), with
the PEFT/HF-compatible target set q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
(torchtitan names wq,wk,wv,wo,w1,w3,w2). At the end of training the adapter is exported in HF PEFT
format (adapter_config.json + adapter_model.safetensors, BF16, HF module names) so vLLM can load it.

Environment knobs (read at config time in every rank):
  LORA_FT_RUN_NAME   run tag (default: config name)
  LORA_FT_OUT        output root for adapter + run json (default /models/lora-ft/runs)
  LORA_FT_DATA       chat-messages JSONL (default /models/lora-ft/data/alpaca_cleaned_chat.jsonl)
  LORA_FT_INT8_GATHER  "int8" (default; all-gather int8 shards) or "bf16" (cast before gather)
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import gc
import json
import os
import re
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.qwen3_5 import model_registry as tpu_qwen35_model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TpuTrainerCompileConfig, TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

__all__ = [
    "qwen38_27b_lora_bf16",
    "qwen38_27b_qlora_int8",
    "qwen38_27b_qlora_nf4",
    "qwen35_debug_lora_bf16",
    "qwen35_debug_qlora_int8",
    "qwen35_debug_qlora_nf4",
]

QWEN38_27B_HF = "/models/models/Qwen/Qwen3.8-27B"
DEFAULT_DATA = "/models/lora-ft/data/alpaca_cleaned_chat.jsonl"

# torchtitan module path suffix -> HF Qwen3.8 (Qwen3_5ForConditionalGeneration) module path suffix.
TT2HF_MODULE = {
    "attn.wq": "self_attn.q_proj",
    "attn.wk": "self_attn.k_proj",
    "attn.wv": "self_attn.v_proj",
    "attn.wo": "self_attn.o_proj",
    "feed_forward.w1": "mlp.gate_proj",
    "feed_forward.w3": "mlp.up_proj",
    "feed_forward.w2": "mlp.down_proj",
    "attn.in_proj_z": "linear_attn.in_proj_z",
    "attn.out_proj": "linear_attn.out_proj",
}
LORA_TARGETS_TT = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]

# Filled by the config function (runs in every rank before the trainer is built).
_RUN: dict[str, Any] = {
    "config": None,
    "quant": None,
    "rank": 16,
    "alpha": 32.0,
    "targets": list(LORA_TARGETS_TT),
    "hf_prefix": "base_model.model.model.language_model.layers",
    "base_model": "Qwen/Qwen3.8-27B",
    "stats": {},
}


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _log(msg: str) -> None:
    if _rank() == 0:
        print(f"[lora-ft] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# HF checkpoint adapter: the Qwen3.8 checkpoint is multimodal (model.language_model.* keys) while
# the TPU lane builds a text-only model (no vision encoder), for which upstream expects model.*.
# --------------------------------------------------------------------------------------------
class Qwen38VLTextOnlyStateDictAdapter(Qwen35StateDictAdapter):
    """Loads the text decoder of the multimodal Qwen3.8 checkpoint into the text-only TPU model."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self.hf_language_model_prefix = "model.language_model"
        self.from_hf_map = {
            (k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k): v
            for k, v in self.from_hf_map.items()
        }


# --------------------------------------------------------------------------------------------
# Dtype policy: frozen base weights are stored in BF16, LoRA adapters keep FP32 master weights.
# --------------------------------------------------------------------------------------------
def _cast_frozen_params_(model: nn.Module, dtype: torch.dtype) -> None:
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if p is None or p.requires_grad or not p.is_floating_point() or p.dtype == dtype:
                continue
            mod._parameters[name] = nn.Parameter(p.data.to(dtype), requires_grad=False)


def _noop_init(_p: torch.Tensor) -> None:
    return None


def _wrap_parallelize(orig_fn):
    def parallelize_with_frozen_bf16(model, *args, **kwargs):
        _cast_frozen_params_(model, torch.bfloat16)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        _RUN["stats"]["trainable_params"] = int(n_tr)
        _RUN["stats"]["total_params"] = int(n_all)
        _log(f"params: total={n_all:,} trainable(LoRA)={n_tr:,} ({100.0 * n_tr / max(n_all, 1):.3f}%)")
        if _RUN.get("skip_frozen_init"):
            # Every frozen base weight is overwritten by the HF load (the loader raises if any is
            # missing), so skip their random init; LoRA adapters keep A~kaiming, B=0.
            n_skip = 0
            for mod in model.modules():
                own = list(mod.named_parameters(recurse=False))
                if own and all(not p.requires_grad for _, p in own) and hasattr(mod, "_param_init"):
                    mod._param_init = {n: _noop_init for n, _ in own}
                    n_skip += 1
            _log(f"skipping random init of {n_skip} frozen modules (overwritten by the HF checkpoint load)")
        return orig_fn(model, *args, **kwargs)

    return parallelize_with_frozen_bf16


# --------------------------------------------------------------------------------------------
# INT8 weight-only frozen base (QLoRA-style).
# --------------------------------------------------------------------------------------------
_INT8_CLS_CACHE: dict[type, type] = {}


def _int8_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    w_q = self.weight  # SimpleFSDP property: all-gathered int8 [out, in]
    y = F.linear(x, w_q.to(x.dtype)) * self.weight_scale.to(x.dtype)
    if self.bias is not None:
        y = y + self.bias
    lora_a = getattr(self, "lora_a", None)
    if lora_a is not None:
        y = y + self._lora_scaling * self.lora_b(lora_a(x))
    return y


def _int8_cls(cls: type) -> type:
    new = _INT8_CLS_CACHE.get(cls)
    if new is None:
        new = type(f"Int8WeightOnly{cls.__name__}", (cls,), {"forward": _int8_linear_forward})
        _INT8_CLS_CACHE[cls] = new
    return new


@torch.no_grad()
def quantize_frozen_linears_int8_(model_parts: list[nn.Module], gather_dtype: str = "int8") -> dict:
    """Replaces every frozen decoder-layer Linear weight by int8 + per-output-channel BF16 scale."""
    from torch.distributed.tensor import DTensor, Shard

    n_mod = 0
    n_param = 0
    max_rel_err = 0.0
    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not isinstance(mod, nn.Linear) or not fqn.startswith("layers."):
                continue
            p = mod._parameters.get("weight")
            if p is None or p.requires_grad or not p.is_floating_point():
                continue
            is_dt = isinstance(p, DTensor)
            local = p.to_local() if is_dt else p.data
            w = local.float()
            shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)] if is_dt else []
            amax = w.abs().amax(dim=1)
            if shard_dims and shard_dims[0] == 1:
                torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX,
                                             group=p.device_mesh.get_group())
            scale = (amax.clamp(min=1e-12) / 127.0).to(torch.bfloat16)
            q = torch.round(w / scale.float()[:, None]).clamp_(-127, 127).to(torch.int8)
            if n_mod < 4:  # quick numerical sanity check on a few layers
                err = ((q.float() * scale.float()[:, None] - w).norm() / w.norm().clamp(min=1e-12)).item()
                max_rel_err = max(max_rel_err, err)
            if is_dt and shard_dims and shard_dims[0] == 0:
                scale_full = DTensor.from_local(scale, p.device_mesh, [Shard(0)], run_check=False,
                                                shape=(p.shape[0],), stride=(1,)).full_tensor()
            else:
                scale_full = scale
            if is_dt:
                new = DTensor.from_local(q, p.device_mesh, p.placements, run_check=False,
                                         shape=p.shape, stride=p.stride())
            else:
                new = q
            mod._parameters["weight"] = nn.Parameter(new, requires_grad=False)
            mod.register_buffer("weight_scale", scale_full.contiguous(), persistent=False)
            if gather_dtype == "int8":
                for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                    rc.param_dtype = None  # all-gather the int8 shard, do not upcast before the gather
            mod.__class__ = _int8_cls(type(mod))
            n_mod += 1
            n_param += p.numel()
    gc.collect()
    return {"int8_modules": n_mod, "int8_params": int(n_param),
            "int8_rel_fro_err_first4": round(max_rel_err, 6), "int8_gather_dtype": gather_dtype}


# --------------------------------------------------------------------------------------------
# NF4 frozen base (closest TPU analogue of QLoRA's storage format).
#   * QLoRA NF4 codebook (16 quantiles of N(0,1), normalized to [-1, 1]), blocks of 64 consecutive
#     weights along the input dim, absmax scaling per block.
#   * Double quantization of the block absmax to 8 bits (code/255 * per-row max, row max kept in BF16;
#     QLoRA uses 8-bit floats with block 256 -> 0.127 bit/param, here 0.125 bit/param + 16 bit/row).
#   * Per Linear one uint8 tensor [out, in/2 + in/64] = packed 4-bit codes | absmax codes, FSDP-sharded
#     on rows and all-gathered as uint8; decoded with a 15-way select tree (no gather), then BF16 matmul.
# --------------------------------------------------------------------------------------------
NF4_CODE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
            -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
            0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0]
NF4_BLOCK = 64
_NF4_CLS_CACHE: dict[type, type] = {}


def _nf4_decode(idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """4-bit codes (any int dtype, values 0..15) -> NF4 values, via a binary select tree."""
    i = idx.to(torch.int32)
    bits = [(i & (1 << k)) != 0 for k in range(4)]
    level = [torch.where(bits[0], NF4_CODE[2 * j + 1], NF4_CODE[2 * j]) for j in range(8)]
    level = [torch.where(bits[1], level[2 * j + 1], level[2 * j]) for j in range(4)]
    level = [torch.where(bits[2], level[2 * j + 1], level[2 * j]) for j in range(2)]
    return torch.where(bits[3], level[1], level[0]).to(dtype)


def _nf4_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    st = self.weight  # SimpleFSDP property: all-gathered uint8 [out, in/2 + in/64]
    out_f, in_f = self.out_features, self.in_features
    nb = in_f // NF4_BLOCK
    packed, acode = st[:, : in_f // 2], st[:, in_f // 2:]
    idx = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(out_f, in_f)
    absmax = (acode.to(torch.float32) * (1.0 / 255.0)) * self.weight_rowmax.to(torch.float32)[:, None]
    w = (_nf4_decode(idx, torch.float32).reshape(out_f, nb, NF4_BLOCK) * absmax[..., None]).reshape(out_f, in_f)
    y = F.linear(x, w.to(x.dtype))
    if self.bias is not None:
        y = y + self.bias
    lora_a = getattr(self, "lora_a", None)
    if lora_a is not None:
        y = y + self._lora_scaling * self.lora_b(lora_a(x))
    return y


def _nf4_cls(cls: type) -> type:
    new = _NF4_CLS_CACHE.get(cls)
    if new is None:
        new = type(f"NF4WeightOnly{cls.__name__}", (cls,), {"forward": _nf4_linear_forward})
        _NF4_CLS_CACHE[cls] = new
    return new


@torch.no_grad()
def quantize_frozen_linears_nf4_(model_parts: list[nn.Module]) -> dict:
    from torch.distributed.tensor import DTensor, Shard

    mids = [(NF4_CODE[j] + NF4_CODE[j + 1]) / 2 for j in range(15)]
    n_mod, n_param, max_rel_err, fallback = 0, 0, 0.0, []
    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not isinstance(mod, nn.Linear) or not fqn.startswith("layers."):
                continue
            p = mod._parameters.get("weight")
            if p is None or p.requires_grad or not p.is_floating_point():
                continue
            is_dt = isinstance(p, DTensor)
            shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)] if is_dt else []
            if (shard_dims and shard_dims[0] != 0) or p.shape[1] % NF4_BLOCK:
                fallback.append(fqn)
                continue
            local = p.to_local() if is_dt else p.data
            w = local.float()
            rows, cols = w.shape
            blocks = w.reshape(rows, cols // NF4_BLOCK, NF4_BLOCK)
            absmax = blocks.abs().amax(-1)
            rowmax = absmax.amax(-1).clamp(min=1e-12).to(torch.bfloat16)
            acode = torch.round(absmax / rowmax.float()[:, None] * 255.0).clamp_(1, 255)
            absmax_dq = acode / 255.0 * rowmax.float()[:, None]
            normed = (blocks / absmax_dq[..., None]).clamp_(-1.0, 1.0)
            idx = torch.zeros(normed.shape, dtype=torch.int32, device=normed.device)
            for m in mids:
                idx += (normed > m).to(torch.int32)
            idx = idx.reshape(rows, cols)
            if n_mod < 4:
                deq = (_nf4_decode(idx, torch.float32).reshape(rows, -1, NF4_BLOCK) * absmax_dq[..., None]).reshape(rows, cols)
                max_rel_err = max(max_rel_err, ((deq - w).norm() / w.norm().clamp(min=1e-12)).item())
            packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(torch.uint8)
            store = torch.cat([packed, acode.to(torch.uint8)], dim=1).contiguous()
            if is_dt and shard_dims:
                rowmax_full = DTensor.from_local(rowmax, p.device_mesh, [Shard(0)], run_check=False,
                                                 shape=(p.shape[0],), stride=(1,)).full_tensor()
            else:
                rowmax_full = rowmax
            if is_dt:
                gshape = (p.shape[0], store.shape[1])
                new = DTensor.from_local(store, p.device_mesh, p.placements, run_check=False,
                                         shape=gshape, stride=(store.shape[1], 1))
            else:
                new = store
            mod._parameters["weight"] = nn.Parameter(new, requires_grad=False)
            mod.register_buffer("weight_rowmax", rowmax_full.contiguous(), persistent=False)
            for rc in getattr(mod, "_simple_fsdp_parametrization_list", []):
                rc.param_dtype = None  # all-gather the uint8 storage as is
            mod.__class__ = _nf4_cls(type(mod))
            n_mod += 1
            n_param += p.numel()
    gc.collect()
    return {"nf4_modules": n_mod, "nf4_params": int(n_param), "nf4_block": NF4_BLOCK,
            "nf4_absmax_double_quant": "uint8 code * rowmax/255 (rowmax bf16)",
            "nf4_rel_fro_err_first4": round(max_rel_err, 6), "nf4_fallback_modules": fallback[:10],
            "bits_per_quantized_param": 4 + 8 / NF4_BLOCK}


def _device_mem_gib() -> dict:
    try:
        from torchtitan.tools.utils import device_module

        st = device_module.memory_stats()
        g = lambda k: round(st.get(k, -1) / 2**30, 3)  # noqa: E731
        return {"allocated_gib": g("allocated_bytes.all.current"), "reserved_gib": g("reserved_bytes.all.current"),
                "peak_allocated_gib": g("allocated_bytes.all.peak"), "peak_reserved_gib": g("reserved_bytes.all.peak")}
    except Exception as e:  # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------------------------
# Direct sharded HF loader. Each rank reads only the rows/cols of its own FSDP shard straight from
# the safetensors files (lazy slices via safe_open) and copies them into its local shard: no DCP
# planning collectives and no DTensor torch.cat for the fused GDN q/k/v (the stock initial_load_in_hf
# path sat >3 min at 0% CPU with zero bytes read on v5p-16 + DEFER_AND_FUSE for this model).
# --------------------------------------------------------------------------------------------
_GDN_SPLIT = {"in_proj_q": ("in_proj_qkv", "q"), "in_proj_k": ("in_proj_qkv", "k"), "in_proj_v": ("in_proj_qkv", "v"),
              "conv_q": ("conv1d", "q"), "conv_k": ("conv1d", "k"), "conv_v": ("conv1d", "v")}


def _hf_source(tt_key: str, adapter, inv_map: dict[str, str]) -> tuple[str, int] | None:
    m = re.match(r"layers\.(\d+)\.(.+)$", tt_key)
    if m is None:
        hf = inv_map.get(tt_key)
        return (hf, 0) if hf else None
    idx, rest = m.group(1), m.group(2)
    parts = rest.split(".")
    if len(parts) == 3 and parts[0] == "attn" and parts[1] in _GDN_SPLIT and parts[2] == "weight":
        fused, which = _GDN_SPLIT[parts[1]]
        dn = adapter.model_config.layers[int(idx)].delta_net
        kd = dn.in_proj_q.out_features
        off = {"q": 0, "k": kd, "v": 2 * kd}[which]
        return f"{adapter.hf_language_model_prefix}.layers.{idx}.linear_attn.{fused}.weight", off
    hf = inv_map.get(f"layers.{{}}.{rest}")
    return (hf.format(idx), 0) if hf else None


@torch.no_grad()
def direct_sharded_hf_load_(model_wrapper, adapter, path: str) -> dict:
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    t0 = time.time()
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    inv_map = {v: k for k, v in adapter.from_hf_map.items() if v is not None}
    handles: dict[str, Any] = {}
    n_loaded, n_bytes, skipped = 0, 0, []
    for part in model_wrapper.model:
        for key, val in part.state_dict().items():
            if ".lora_a." in key or ".lora_b." in key:
                continue  # adapters keep their LoRA init (A ~ kaiming, B = 0)
            src = _hf_source(key, adapter, inv_map)
            if src is None or src[0] not in weight_map:
                skipped.append(key)
                continue
            hf_key, row_off = src
            fname = weight_map[hf_key]
            if fname not in handles:
                handles[fname] = safe_open(os.path.join(path, fname), framework="pt")
            sl = handles[fname].get_slice(hf_key)
            if isinstance(val, DTensor):
                lshape, goff = compute_local_shape_and_global_offset(val.shape, val.device_mesh, val.placements)
                local = val._local_tensor
            else:
                lshape, goff, local = tuple(val.shape), (0,) * val.dim(), val
            idx = tuple(slice(goff[d] + (row_off if d == 0 else 0), goff[d] + (row_off if d == 0 else 0) + lshape[d])
                        for d in range(len(lshape)))
            t = sl[idx] if len(idx) else sl[:]
            if tuple(t.shape) != tuple(local.shape):
                raise RuntimeError(f"shape mismatch for {key} <- {hf_key}{idx}: {tuple(t.shape)} vs {tuple(local.shape)}")
            local.copy_(t.to(dtype=local.dtype).to(local.device))
            n_loaded += 1
            n_bytes += t.numel() * t.element_size()
    params_skipped = [k for k in skipped if not k.endswith(("freqs_cis", "inv_freq", "cache"))]
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(wait=True)
    except Exception:
        pass
    model_wrapper.cached_state_dict = model_wrapper._get_state_dict()
    info = {"loader": "direct_sharded_safetensors", "tensors_loaded": n_loaded,
            "bytes_read_this_rank": int(n_bytes), "seconds": round(time.time() - t0, 1),
            "skipped_keys": skipped[:20], "num_skipped": len(skipped)}
    _log(f"direct HF load: {info}")
    if params_skipped:
        raise RuntimeError(f"HF checkpoint is missing model keys (would train on random weights): {params_skipped[:10]}")
    return info


# --------------------------------------------------------------------------------------------
# Hooks: post-checkpoint-load quantization, and PEFT export after training.
# --------------------------------------------------------------------------------------------
def _full(p: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return p.full_tensor() if isinstance(p, DTensor) else p


@torch.no_grad()
def export_peft_adapter(model_parts: list[nn.Module], out_dir: str) -> dict:
    tensors: dict[str, torch.Tensor] = {}
    hf_targets: set[str] = set()
    n = 0
    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not (hasattr(mod, "lora_a") and hasattr(mod, "lora_b")):
                continue
            clean = fqn.replace("_checkpoint_wrapped_module.", "")
            m = re.match(r"layers\.(\d+)\.(.+)$", clean)
            if m is None or m.group(2) not in TT2HF_MODULE:
                raise RuntimeError(f"cannot map LoRA module {fqn} to an HF name")
            hf_mod = TT2HF_MODULE[m.group(2)]
            hf_targets.add(hf_mod.rsplit(".", 1)[-1])
            prefix = f"{_RUN['hf_prefix']}.{m.group(1)}.{hf_mod}"
            a = _full(mod.lora_a._parameters["weight"]).to(torch.bfloat16)
            b = _full(mod.lora_b._parameters["weight"]).to(torch.bfloat16)
            if _rank() == 0:
                tensors[f"{prefix}.lora_A.weight"] = a.cpu().contiguous()
                tensors[f"{prefix}.lora_B.weight"] = b.cpu().contiguous()
            n += 1
    info = {"lora_modules": n, "hf_target_modules": sorted(hf_targets)}
    if _rank() == 0:
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
        cfg = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": _RUN["base_model"],
            "r": int(_RUN["rank"]), "lora_alpha": float(_RUN["alpha"]), "lora_dropout": 0.0, "bias": "none",
            "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
            "target_modules": sorted(hf_targets), "modules_to_save": None, "use_rslora": False,
            "use_dora": False, "rank_pattern": {}, "alpha_pattern": {}, "layers_to_transform": None,
            "layers_pattern": None, "revision": None,
        }
        with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        lora_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        info.update({"adapter_dir": out_dir, "adapter_tensors": len(tensors), "adapter_bytes": int(lora_bytes)})
    return info


def _install_patches() -> None:
    from torchtitan import trainer as tt_trainer
    from torchtitan.components.checkpointer import base as ckpt_base

    if getattr(ckpt_base.BaseCheckpointManager, "_lora_ft_patched", False):
        return
    try:  # `kill -USR1 <pid>` dumps all Python thread stacks into the run log (hang debugging)
        import faulthandler
        import signal

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass
    orig_load = ckpt_base.BaseCheckpointManager.load

    def load_then_quantize(self, step: int = -1) -> bool:
        t0 = time.time()
        if (self.enable and self.initial_load_path and self.initial_load_in_hf
                and os.environ.get("LORA_FT_HF_LOADER", "direct") == "direct"):
            info = direct_sharded_hf_load_(self.states[ckpt_base.MODEL], self.sd_adapter, self.initial_load_path)
            _RUN["stats"]["hf_load"] = info
            ok = True
        else:
            ok = orig_load(self, step)
        _RUN["stats"]["checkpoint_load_s"] = round(time.time() - t0, 1)
        _RUN["stats"]["mem_after_load"] = _device_mem_gib()
        if _RUN["quant"] in ("int8", "nf4") and self.enable:
            t1 = time.time()
            wrapper = self.states[ckpt_base.MODEL]
            if _RUN["quant"] == "int8":
                info = quantize_frozen_linears_int8_(wrapper.model, os.environ.get("LORA_FT_INT8_GATHER", "int8"))
            else:
                info = quantize_frozen_linears_nf4_(wrapper.model)
            # Drop the checkpointer's cached references to the BF16 tensors so their HBM is freed.
            wrapper.cached_state_dict = wrapper._get_state_dict()
            gc.collect()
            try:
                from torch_tpu._internal.sync import synchronize

                synchronize(wait=True)
            except Exception:
                pass
            info["quantize_s"] = round(time.time() - t1, 1)
            _RUN["stats"].update(info)
            _RUN["stats"]["mem_after_quant"] = _device_mem_gib()
            _log(f"{_RUN['quant']} weight-only base: {info}")
        _log(f"checkpoint loaded in {_RUN['stats']['checkpoint_load_s']}s; mem={_RUN['stats'].get('mem_after_quant', _RUN['stats']['mem_after_load'])}")
        return ok

    ckpt_base.BaseCheckpointManager.load = load_then_quantize
    ckpt_base.BaseCheckpointManager._lora_ft_patched = True

    orig_train = tt_trainer.Trainer.train

    def train_then_export(self):
        t0 = time.time()
        out = orig_train(self)
        _RUN["stats"]["train_wall_s"] = round(time.time() - t0, 1)
        _RUN["stats"]["mem_end"] = _device_mem_gib()
        run = os.environ.get("LORA_FT_RUN_NAME", _RUN["config"])
        ts = os.environ.get("LORA_FT_TS", _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        root = os.path.join(os.environ.get("LORA_FT_OUT", "/models/lora-ft/runs"), f"{run}_{ts}")
        try:
            t1 = time.time()
            info = export_peft_adapter(self.model_parts, os.path.join(root, "peft_adapter"))
            info["export_s"] = round(time.time() - t1, 1)
            _RUN["stats"]["export"] = info
        except Exception as e:  # keep training results even if export fails
            _RUN["stats"]["export"] = {"error": f"{type(e).__name__}: {e}"}
        if _rank() == 0:
            os.makedirs(root, exist_ok=True)
            rec = {"config": _RUN["config"], "quant": _RUN["quant"], "lora_rank": _RUN["rank"],
                   "lora_alpha": _RUN["alpha"], "lora_targets_tt": _RUN["targets"],
                   "world_size": int(os.environ.get("WORLD_SIZE", "1")), **_RUN["stats"]}
            with open(os.path.join(root, "run_stats.json"), "w") as f:
                json.dump(rec, f, indent=2)
            _log(f"run stats -> {root}/run_stats.json : {json.dumps(rec)}")
        return out

    tt_trainer.Trainer.train = train_then_export


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return sample["messages"]


def _is_not_none(sample: Any) -> bool:
    return sample is not None


def sft_dataloader(path: str | None = None, seed: int = 42) -> GrainDataLoader.Config:
    ds = SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json", split="train",
            load_dataset_kwargs={"data_files": path or os.environ.get("LORA_FT_DATA", DEFAULT_DATA)},
        ),
        processor=ChatProcessor.Config(messages_fn=chat_messages),
        post_filters=(_is_not_none,),
    )
    return GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=ds), seed=seed, shuffle=True)


# --------------------------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------------------------
def _lora_spec(flavor: str, seq_len: int, rank: int, alpha: float, targets: list[str], hf_adapter: bool):
    spec = tpu_qwen35_model_registry(flavor, seq_len=seq_len)
    model_cfg = LoRAConverter.Config(rank=rank, alpha=alpha, target_modules=list(targets)).build().convert(spec.model)
    kw = {"model": model_cfg, "parallelize_fn": _wrap_parallelize(spec.parallelize_fn)}
    if hf_adapter:
        kw["state_dict_adapter"] = Qwen38VLTextOnlyStateDictAdapter
    return dataclasses.replace(spec, **kw)


def _base_config(*, name: str, quant: str | None, flavor: str = "27B", seq_len: int = 4096,
                 seqs_per_rank: int = 2, steps: int = 150, lr: float = 2e-4, rank: int = 16,
                 alpha: float = 32.0, hf_path: str | None = QWEN38_27B_HF,
                 data_path: str | None = None) -> TPUTrainerConfig:
    _RUN.update({"config": name, "quant": quant, "rank": rank, "alpha": alpha, "targets": list(LORA_TARGETS_TT),
                 "skip_frozen_init": hf_path is not None and os.environ.get("LORA_FT_SKIP_FROZEN_INIT", "0") == "1"})
    _install_patches()
    cfg = TPUTrainerConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=hf_path or "./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=_lora_spec(flavor, seq_len, rank, alpha, LORA_TARGETS_TT, hf_adapter=hf_path is not None),
        optimizer=default_adamw(lr=lr, weight_decay=0.0, implementation="foreach"),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10, decay_ratio=0.9, decay_type="cosine",
                                                  min_lr_factor=0.1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seqs_per_rank * seq_len,
            max_context_length=seq_len,
            steps=steps,
            max_norm=1.0,
            dtype="float32",  # LoRA master weights + AdamW state; frozen base is cast to BF16
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        dataloader=sft_dataloader(data_path) if hf_path else _c4_test_loader(),
        parallelism=ParallelismConfig(use_simple_fsdp=True, data_parallel_shard_degree=-1,
                                      fsdp_reshard_after_forward="always"),
        comm=CommConfig(init_timeout_seconds=3600, train_timeout_seconds=3600),
        checkpoint=CheckpointManager.Config(
            # Always enabled (load_only) so the post-load hook runs; with hf_path=None it is a fresh
            # random init (debug smoke tests). Full checkpoints are never written; the adapter is
            # exported in PEFT format by the train hook.
            enable=True,
            load_only=True,
            initial_load_path=hf_path,
            initial_load_in_hf=hf_path is not None,
            initial_load_model_only=hf_path is not None,
        ),
        activation_checkpoint=FullAC.Config(preserve_rng_state=False),
        compile=TpuTrainerCompileConfig(enable=False, backend="tpu", compile_grad_norm=True),
        validator=Validator.Config(enable=False),
        tpu_config=TPUConfig(enable_amp=False, eager_mode="DEFER_AND_FUSE"),
    )
    cfg.debug.seed = 42
    return cfg


def _c4_test_loader():
    from torchtitan.experiments.tpu.datasets import c4_test_dataloader

    return c4_test_dataloader()


def qwen38_27b_lora_bf16() -> TPUTrainerConfig:
    """Qwen3.8-27B LoRA (r16/a32, q,k,v,o,gate,up,down), frozen BF16 base, SFT on alpaca-cleaned."""
    return _base_config(name="qwen38_27b_lora_bf16", quant=None)


def qwen38_27b_qlora_int8() -> TPUTrainerConfig:
    """Qwen3.8-27B QLoRA-style: same adapters on an int8 weight-only (per-channel) frozen base."""
    return _base_config(name="qwen38_27b_qlora_int8", quant="int8")


def qwen35_debug_lora_bf16() -> TPUTrainerConfig:
    """Tiny smoke test (random-init Qwen3.5 debugmodel, c4_test, no checkpoint)."""
    _RUN["hf_prefix"] = "base_model.model.model.layers"
    return _base_config(name="qwen35_debug_lora_bf16", quant=None, flavor="debugmodel", seq_len=256,
                        seqs_per_rank=8, steps=10, hf_path=None)


def qwen35_debug_qlora_int8() -> TPUTrainerConfig:
    """Tiny smoke test of the int8 path; quantization needs a checkpoint load, so it runs on random weights."""
    _RUN["hf_prefix"] = "base_model.model.model.layers"
    return _base_config(name="qwen35_debug_qlora_int8", quant="int8", flavor="debugmodel", seq_len=256,
                        seqs_per_rank=8, steps=10, hf_path=None)


def qwen38_27b_qlora_nf4() -> TPUTrainerConfig:
    """Qwen3.8-27B QLoRA-format: same adapters on an NF4 (block 64, double-quantized absmax) frozen base."""
    return _base_config(name="qwen38_27b_qlora_nf4", quant="nf4")


def qwen35_debug_qlora_nf4() -> TPUTrainerConfig:
    """Tiny smoke test of the NF4 path (random weights)."""
    _RUN["hf_prefix"] = "base_model.model.model.layers"
    return _base_config(name="qwen35_debug_qlora_nf4", quant="nf4", flavor="debugmodel", seq_len=256,
                        seqs_per_rank=8, steps=10, hf_path=None)
