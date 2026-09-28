"""
Shared utilities.
"""

import os
import re
import warnings
import argparse

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from baukit import TraceDict
from sklearn.metrics import f1_score
from custom_llama import llama


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_hf_token():
    """Use HF_TOKEN or the locally cached Hugging Face login."""
    return os.environ.get("HF_TOKEN") or None


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser,
                    token_position_choices=("last", "mean")) -> None:
    """Add the args that every pipeline script shares."""
    parser.add_argument('--model', type=str,
                        default='NousResearch/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--base-dir', type=str, default='./data',
                        help='Base data directory, e.g. ./data')
    parser.add_argument('--category', type=str, default='ability',
                        help='Category folder, e.g. ability, religion, politics')
    parser.add_argument('--template-type', type=str,
                        choices=['listing', 'definition', 'simple'],
                        default='listing')
    parser.add_argument('--templates-per-side', type=int, default=30)
    parser.add_argument('--token-position', type=str,
                        choices=list(token_position_choices),
                        default='mean')


def resolve_paths(args) -> tuple[str, str]:
    """Build the same DATA_DIR / RESULTS_DIR everywhere."""
    experiment_folder = (
        f"{args.template_type}_n{args.templates_per_side}_{args.token_position}"
    )
    data_dir = os.path.join(args.base_dir, args.category, experiment_folder)
    results_dir = os.path.join(data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_tokenizer(model_name: str, use_fast: bool = True):
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='./model',
        token=get_hf_token(),
        use_fast=use_fast,
    )


def load_model(model_name: str, device_map: str | None = None, dtype=None):
    """Load the custom Llama. Pass device_map='auto' for multi-GPU/sharded.

    dtype defaults to bfloat16 (override with the PIPELINE_DTYPE env var:
    bfloat16 | float16 | float32). float16 is NOT recommended for Llama-3:
    its activations can exceed the float16 range and become inf/NaN.
    """
    if dtype is None:
        dtype = os.environ.get("PIPELINE_DTYPE", "bfloat16")
    if isinstance(dtype, str):
        dtype = _DTYPES[dtype]
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("WARNING: GPU has no bfloat16 support; falling back to float32")
        dtype = torch.float32
    print(f"Loading {model_name} in {dtype}")
    kwargs = dict(
        cache_dir='./model',
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        token=get_hf_token(),
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
        return llama.LlamaForCausalLM.from_pretrained(model_name, **kwargs)
    return llama.LlamaForCausalLM.from_pretrained(model_name, **kwargs).to('cuda:0')


def _input_device(model):
    """Device of the embedding layer (correct also with device_map='auto')."""
    return model.get_input_embeddings().weight.device


def _check_finite(rep, idx, prompt, tokenizer, model, kind, on_nonfinite):
    """rep: (num_layers, dim). Report prompt + layers if anything is NaN/inf."""
    finite_rows = np.isfinite(rep).all(axis=-1)
    if finite_rows.all():
        return
    bad_layers = np.flatnonzero(~finite_rows).tolist()
    try:
        text = tokenizer.decode(prompt[0])
    except Exception:
        text = "<undecodable>"
    msg = (f"Non-finite {kind} activations for prompt #{idx} {text!r} in layers "
           f"{bad_layers} (model dtype {model.dtype}). If the model is float16, "
           f"load it in bfloat16 or float32.")
    if on_nonfinite == "raise":
        raise FloatingPointError(msg)
    print("WARNING: " + msg)   # print, not warnings.warn: callers silence warnings


# ---------------------------------------------------------------------------
# Token-finding
# ---------------------------------------------------------------------------

def find_token_indices(input_ids, word, tokenizer, on_missing="raise",
                       max_seq_len: int = 6):
    """Locate `word` in tokenized `input_ids`.

    Tries spans of length 1..max_seq_len. Strips whitespace and BPE markers
    before comparing. Returns the list of token indices that make up the word.
    """
    decoded = [tokenizer.decode([tid]).strip() for tid in input_ids[0]]
    word_clean = re.sub(r"\s+", "", word.lower())

    for seq_len in range(1, max_seq_len + 1):
        for start in range(len(decoded) - seq_len + 1):
            joined = "".join(decoded[start:start + seq_len]).strip()
            joined = joined.replace("▁", "").replace("<0x", "").replace(">", "")
            joined_clean = re.sub(r"\s+", "", joined.lower())
            if joined_clean == word_clean:
                return list(range(start, start + seq_len))

    if on_missing == "raise":
        raise ValueError(
            f"Could not locate '{word}' in tokenized sentence: {decoded}. "
            f"This indicates a tokenization mismatch."
        )
    print(f"Warning: could not find '{word}', using last token")
    return [len(decoded) - 1]


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_layer_representations(model, tokenizer, statements,
                                  target_words=None,
                                  token_position: str = "mean",
                                  on_nonfinite: str = "warn"):
    """Extract residual-stream representations from `model.layers.{i}`.

    If `target_words` is provided, extract at the target word token (averaged
    over its sub-tokens). Otherwise honor `token_position` ('mean' or 'last').

    Returns:
        float32 features array shaped (N, 1, num_layers, 1, hidden_dim) — the
        extra singleton axes preserve compatibility with the head-level
        pipeline's downstream indexing.
    """
    if target_words is None and token_position not in ("mean", "last"):
        raise ValueError(f"Unknown token_position: {token_position}")
    LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    device = _input_device(model)
    layer_wise_hidden_states_list = []

    for idx, prompt in tqdm(list(enumerate(statements)), total=len(statements)):
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.tensor(prompt)
        prompt = prompt.to(device)

        token_indices = None
        if target_words is not None:
            token_indices = find_token_indices(prompt, target_words[idx], tokenizer)

        with torch.no_grad():
            with TraceDict(model, LAYERS) as ret:
                _ = model(prompt)

                layer_hidden_states = []
                for layer_name in LAYERS:
                    layer_output = ret[layer_name].output
                    if isinstance(layer_output, tuple):
                        layer_output = layer_output[0]
                    # upcast BEFORE numpy / pooling (bf16 has no numpy dtype)
                    layer_output = layer_output.squeeze(0).detach().float().cpu().numpy()

                    if token_indices is not None:
                        rep = layer_output[token_indices].mean(axis=0)
                    elif token_position == "mean":
                        rep = layer_output.mean(axis=0)
                    else:
                        rep = layer_output[-1]
                    layer_hidden_states.append(rep)

        layer_hidden_states = np.stack(layer_hidden_states, axis=0)
        _check_finite(layer_hidden_states, idx, prompt, tokenizer, model,
                      "residual-stream", on_nonfinite)
        layer_wise_hidden_states_list.append(layer_hidden_states)

    features = np.stack(layer_wise_hidden_states_list, axis=0)
    features = features[:, np.newaxis, :, np.newaxis, :]
    print(f"Features shape: {features.shape}")
    return features


def extract_attention_head_activations(model, tokenizer, statements,
                                       token_position: str = "mean",
                                       target_words=None,
                                       on_nonfinite: str = "warn"):
    """Extract per-head attention output from `self_attn.head_out` hooks.

    `token_position` is one of 'mean', 'last', or 'word'.
    'word' requires `target_words` (parallel list, one entry per prompt).

    Tokens are pooled right after each forward pass, so only one (L, D) vector
    per prompt is kept in memory instead of the full (L, T, D) states.

    Returns:
        float32 features array shaped (N, 1, num_layers, num_heads, head_dim).
    """
    if token_position not in ("mean", "last", "word"):
        raise ValueError(f"Unknown token_position: {token_position}")
    if token_position == "word" and target_words is None:
        raise ValueError("target_words must be provided when token_position='word'.")

    HEADS = [f"model.layers.{i}.self_attn.head_out"
             for i in range(model.config.num_hidden_layers)]
    num_heads = model.config.num_attention_heads
    device = _input_device(model)
    features = []

    for idx, prompt in enumerate(tqdm(statements, total=len(statements))):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                _ = model(prompt.to(device))
                # squeeze(0) only — plain squeeze() would also collapse the
                # token dim for single-token prompts and corrupt shapes.
                # .float() BEFORE .numpy(): pool in float32, and bf16 has no numpy dtype.
                hs = torch.stack(
                    [ret[h].output.squeeze(0).detach().float().cpu() for h in HEADS],
                    dim=0,
                ).numpy()                                   # (L, T, H*D)

        if token_position == "mean":
            rep = hs.mean(axis=1)
        elif token_position == "last":
            rep = hs[:, -1, :]
        else:
            token_indices = find_token_indices(prompt, target_words[idx], tokenizer)
            rep = hs[:, token_indices, :].mean(axis=1)

        _check_finite(rep, idx, prompt, tokenizer, model, "attention-head", on_nonfinite)
        features.append(rep.reshape(1, rep.shape[0], num_heads, -1))   # (1, L, H, D)

    return np.stack(features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Direction / projection / accuracy
# ---------------------------------------------------------------------------

def find_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class-mean difference direction (no normalization — magnitude kept)."""
    X = np.asarray(X, dtype=np.float64)
    pos_mask = y == 1
    neg_mask = y == -1
    if not pos_mask.any() or not neg_mask.any():
        raise ValueError("find_direction needs both +1 and -1 examples")
    pos_mean = np.mean(X[pos_mask], axis=0)
    neg_mean = np.mean(X[neg_mask], axis=0)
    return pos_mean - neg_mean


def normalize_projections(projections, reference=None, normalization="minmax"):
    """Fit scaling on reference projections, or on projections if omitted."""
    if normalization not in ("minmax", "zscore"):
        raise ValueError("normalization must be 'minmax' or 'zscore'")
    ref = projections if reference is None else reference
    if normalization == "zscore":
        return (projections - ref.mean()) / (ref.std() + 1e-8)
    return 2 * (projections - ref.min()) / (ref.max() - ref.min() + 1e-8) - 1


def find_projection(X: np.ndarray, direction: np.ndarray,
                    rescale: bool = True, normalization: str = "minmax") -> np.ndarray:
    """Project, then optionally normalize using this population's statistics."""
    projections = X @ direction / (direction @ direction + 1e-8)
    return normalize_projections(projections, normalization=normalization) if rescale else projections


def find_projection_joint(X: np.ndarray, reference: np.ndarray,
                          direction: np.ndarray, rescale: bool = True,
                          normalization: str = "minmax") -> np.ndarray:
    """Project and optionally normalize using a supplied calibration population."""
    denom = direction @ direction + 1e-8
    projections = X @ direction / denom
    if not rescale:
        return projections
    return normalize_projections(projections, reference @ direction / denom, normalization)


def macro_f1(projections, true_labels, threshold: float = 0.0) -> float:
    """Macro-F1: average of per-class F1 between {-1, +1} predictions and labels.

    Class-imbalance robust counterpart to `accuracy`. Computes F1 separately
    for the positive class (+1) and the negative class (-1), then averages.
    Returns 0.0 when called with empty inputs, mirroring `accuracy`.
    """

    proj = np.asarray(projections).reshape(-1)
    y_true = np.asarray(true_labels).reshape(-1)
    if proj.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"macro_f1(): length mismatch proj={proj.shape[0]} "
            f"vs labels={y_true.shape[0]}"
        )
    if y_true.shape[0] == 0:
        return 0.0
    preds = np.where(proj > threshold, 1, -1)
    return float(f1_score(y_true.astype(int), preds, average="macro",
                          labels=[-1, 1], zero_division=0))


def accuracy(projections, true_labels, threshold: float = 0.0) -> float:
    """proj > threshold -> +1, else -> -1."""
    proj = np.asarray(projections).reshape(-1)
    y_true = np.asarray(true_labels).reshape(-1)
    if proj.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"accuracy(): length mismatch proj={proj.shape[0]} vs labels={y_true.shape[0]}"
        )
    if y_true.shape[0] == 0:
        return 0.0
    preds = np.where(proj > threshold, 1, -1)
    return float((preds == y_true.astype(int)).mean())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_axes_json(data_dir: str) -> dict:
    """Load the antonym-pairs JSON used by every script."""
    import json
    path = os.path.join(
        data_dir,
        "antonym_pairs_by_identifier_per_side_people_authomat.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def axis_line(entry: dict) -> str:
    """Extract the canonical 'pos|neg' axis label from an entry."""
    line = entry["antonym_pairs"].split("\n")[0]
    return line.replace("Antonym Pair: ", "")


def iter_axis_examples(entry: dict):
    """Yield (sentence, label, adjective_word) for every example in an axis.

    Raises if the dataset wasn't regenerated with the third (adjective) field.
    """
    for source, pairs in entry["examples"].items():
        for item in pairs:
            if len(item) < 3:
                raise ValueError(
                    f"Expected [sentence, label, adjective_adj] in JSON examples, "
                    f"got: {item}. Re-run 2-0.py to regenerate the dataset."
                )
            sent, label, adj = item[0], item[1], item[2]
            if not sent.strip():
                continue
            yield sent.strip(), label, adj


def read_target_words(path: str, per_axis: bool = False):
    """Load a target-words file in one of two formats.

    Legacy format (per_axis=False, default):
        Tab-separated  qid<TAB>name[<TAB>label]
        One label per word, used across all axes.
        Returns:
            words:  list[(qid, name)]
            labels: list[(qid, name, label)] — rows that had a third column

    Per-axis format (per_axis=True):
        Tab-separated  qid<TAB>name<TAB>pair<TAB>label
        Labels depend on the axis (`pair`), e.g. "unfriendly|friendly".
        Returns:
            words:          list[(qid, name)] — deduped (preserves first-seen order)
            labels_by_axis: dict[pair -> list[(qid, name, label)]]
    """
    if not per_axis:
        words = []
        labels = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    qid, name = parts[0], parts[1]
                    words.append((qid, name))
                    if len(parts) >= 3:
                        labels.append((qid, name, int(parts[2])))

        print(f"Read {len(words)} words and {len(labels)} labels from {path}")
        if len(labels) == 0:
            print("WARNING: No labels were read!")
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"Line {i+1}: {repr(line)}")
                    else:
                        break
        return words, labels

    # per_axis=True: one row per (axis, word) with its own label.
    seen_qids = set()
    words = []
    labels_by_axis: dict[str, list[tuple[str, str, int]]] = {}
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"per_axis=True expects 4 tab-separated columns "
                    f"(qid, name, pair, label); got {len(parts)} in: {line!r}"
                )
            qid, name, pair, lab = parts[0], parts[1], parts[2], int(parts[3])
            if qid not in seen_qids:
                seen_qids.add(qid)
                words.append((qid, name))
            labels_by_axis.setdefault(pair, []).append((qid, name, lab))
            n_rows += 1

    print(f"Read {n_rows} (axis, word) labels across {len(labels_by_axis)} axes "
          f"and {len(words)} unique words from {path}")
    return words, labels_by_axis


def filter_for_axis(features: np.ndarray,
                    words: list[tuple[str, str]],
                    labels_for_axis: list[tuple[str, str, int]]):
    """Select rows of `features` matching this axis's labeled words.

    Used together with `read_target_words(path, per_axis=True)` so each axis
    can pull its own labeled subset of the full target-word feature matrix.

    Args:
        features:        shape (N, ...) with N matching len(words).
        words:           list[(qid, name)] returned by read_target_words.
        labels_for_axis: list[(qid, name, label)] for one axis.

    Returns:
        features_subset: features rows for this axis's labeled words, in the
                         same order as labels_for_axis.
        y_true:          int array of labels in the same order.
    """
    qid_to_idx = {qid: i for i, (qid, _) in enumerate(words)}
    indices, y = [], []
    for qid, _name, lab in labels_for_axis:
        if qid not in qid_to_idx:
            continue
        indices.append(qid_to_idx[qid])
        y.append(int(lab))
    if not indices:
        return features[:0], np.zeros(0, dtype=int)
    return features[indices], np.asarray(y, dtype=int)
