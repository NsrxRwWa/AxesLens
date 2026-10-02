"""Activation extraction for stock Hugging Face models.
Needs a transformers version that knows the model (Qwen3.5: transformers >= 5).
"""
import os

import numpy as np
import torch

LEVELS = ("head", "linhead", "layer")
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ----------------------------------------------------------------------------- loading
def hf_token():
    return os.environ.get("HF_TOKEN") or None


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, token=hf_token())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # causal model: real tokens never see the padding
    return tok


def load_model(name, dtype=None, device_map=None):
    """Text-only causal LM (for Qwen3.5 the vision tower is not loaded).

    Fails loudly if any language-model weight is missing, because missing weights would
    be randomly initialised and silently corrupt every activation.
    """
    from transformers import AutoModelForCausalLM
    dtype = dtype or os.environ.get("PIPELINE_DTYPE", "bfloat16")
    dtype = _DTYPES[dtype] if isinstance(dtype, str) else dtype
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("WARNING: GPU has no bfloat16 support; falling back to float32")
        dtype = torch.float32
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {name} in {dtype} on {device_map}")
    model, info = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map=device_map, token=hf_token(),
        low_cpu_mem_usage=True, output_loading_info=True)
    if info.get("missing_keys"):
        raise RuntimeError(f"{len(info['missing_keys'])} weights missing when loading {name}, "
                           f"e.g. {sorted(info['missing_keys'])[:5]}")
    model.eval()
    return model


def input_device(model):
    return model.get_input_embeddings().weight.device


# ----------------------------------------------------------------------------- where to read
def find_decoder_layers(model):
    """(dotted path, ModuleList) of the language model's decoder layers."""
    n = model.config.get_text_config().num_hidden_layers
    found = [(name, m) for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n and name.split(".")[-1] == "layers"]
    if len(found) != 1:
        raise RuntimeError(f"Expected one decoder-layer list of length {n}, found {[f[0] for f in found]}")
    return found[0]


def tracepoints(model):
    """{level: [(layer index, module path, module, n_heads, head_dim)]} for this model.

    Heads are read from the input of the output projection, whose width is
    n_heads * head_dim; head_dim comes from the attention module itself.
    """
    cfg = model.config.get_text_config()
    prefix, layers = find_decoder_layers(model)
    points = {lv: [] for lv in LEVELS}
    for i, layer in enumerate(layers):
        points["layer"].append((i, f"{prefix}.{i}", layer, 1, cfg.hidden_size))
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "o_proj"):
            hd = getattr(attn, "head_dim", None) or getattr(cfg, "head_dim", None) \
                or cfg.hidden_size // cfg.num_attention_heads
            points["head"].append((i, f"{prefix}.{i}.self_attn.o_proj", attn.o_proj,
                                   attn.o_proj.in_features // hd, hd))
        lin = getattr(layer, "linear_attn", None)
        if lin is not None and hasattr(lin, "out_proj"):
            hd = getattr(lin, "head_v_dim", None) or cfg.linear_value_head_dim
            points["linhead"].append((i, f"{prefix}.{i}.linear_attn.out_proj", lin.out_proj,
                                      lin.out_proj.in_features // hd, hd))
    for lv, pts in points.items():
        for (i, path, mod, h, d) in pts:
            if lv != "layer" and mod.in_features != h * d:
                raise RuntimeError(f"{path}: in_features {mod.in_features} != {h} heads x {d}")
        if len({(h, d) for *_, h, d in pts}) > 1:
            raise RuntimeError(f"Level {lv}: layers differ in (n_heads, head_dim)")
    return {lv: pts for lv, pts in points.items() if pts}


def describe(points):
    """{level: {"layers": [...], "n_heads": H, "head_dim": D}} (saved next to activations)."""
    return {lv: {"layers": [p[0] for p in pts], "n_heads": pts[0][3], "head_dim": pts[0][4]}
            for lv, pts in points.items()}


# ----------------------------------------------------------------------------- extraction
class PooledActivations:
    """Forward hooks that pool every tracepoint over tokens right away (mean over real
    tokens, and the last real token), so the full (B, T, D) states are never stored."""

    def __init__(self, model, points, positions=("mean", "last")):
        self.points, self.positions = points, positions
        self.handles, self.buf, self.active = [], {}, False
        for lv, pts in points.items():
            for j, (_, _, mod, _, _) in enumerate(pts):
                self.handles.append(mod.register_forward_hook(self._hook(lv, j)))

    def _hook(self, level, j):
        def fn(module, inputs, output):
            if not self.active:                       # model called outside __call__
                return
            x = inputs[0] if level != "layer" else (output[0] if isinstance(output, tuple) else output)
            x = x.float()                                                    # (B, T, D)
            if "mean" in self.positions:
                self.buf[(level, "mean", j)] = (x * self.mask).sum(1) / self.mask.sum(1)
            if "last" in self.positions:
                self.buf[(level, "last", j)] = x[self.rows, self.last]
        return fn

    def __call__(self, model, enc):
        """Run one batch; returns {f"{level}_{pos}": float32 array (B, L_level, n_heads*head_dim)}."""
        am = enc["attention_mask"]
        self.mask = am[:, :, None].float()
        self.last = am.sum(1) - 1
        self.rows = torch.arange(am.shape[0], device=am.device)
        self.buf, self.active = {}, True
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            self.active = False
        out = {}
        for lv, pts in self.points.items():
            for pos in self.positions:
                a = torch.stack([self.buf[(lv, pos, j)] for j in range(len(pts))], 1).cpu().numpy()
                if not np.isfinite(a).all():
                    raise FloatingPointError(f"Non-finite activations ({lv}, {pos})")
                out[f"{lv}_{pos}"] = a
        return out

    def close(self):
        for h in self.handles:
            h.remove()
