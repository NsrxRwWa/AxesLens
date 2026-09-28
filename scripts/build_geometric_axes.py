"""Find a direction (geometric axis) for every semantic axis, per attention head and per layer.

One forward pass per sentence gives all four results at once:
  head_mean, head_last    (attention-head outputs, 32 x 32 heads)
  layer_mean, layer_last  (residual stream, 32 layers; the layer-level baseline)
For each axis and each head/layer it stores:
  score     = between-class / within-class variance ratio
  direction = mean(positive sentences) - mean(negative sentences)

Usage (Colab):
  python 2-theta.py --model NousResearch/Meta-Llama-3-8B-Instruct \
      --data probing_listing_n30.json \
      --out-dir /content/drive/MyDrive/axeslens_checkpoints \
      --hf-repo <user>/axeslens-directions

Checkpoints: results are saved every --chunk-size axes; a restarted run skips finished
chunks. At the end the chunks are merged into four .npz files, uploaded to the
Hugging Face dataset repo (if --hf-repo is given), and the local copies are removed.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from baukit import TraceDict
from tqdm.auto import tqdm

import pipeline_utils as pu

OUTPUTS = ["head_mean", "head_last", "layer_mean", "layer_last"]


def extract(model, tokenizer, sentences, batch_size):
    """Mean- and last-token activations of every head and layer, for all sentences.

    Padding is added on the right: with causal attention, real tokens never see the
    padding, so batching does not change their activations.
    """
    n_layers = model.config.num_hidden_layers
    heads = [f"model.layers.{i}.self_attn.head_out" for i in range(n_layers)]
    layers = [f"model.layers.{i}" for i in range(n_layers)]
    device = model.get_input_embeddings().weight.device
    out = {k: [] for k in OUTPUTS}

    for start in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[start:start + batch_size], return_tensors="pt",
                        padding=True).to(device)
        mask = enc["attention_mask"][:, None, :, None].float()     # (B, 1, T, 1)
        last = enc["attention_mask"].sum(1) - 1                     # index of last real token
        rows = torch.arange(len(last), device=device)
        with torch.no_grad(), TraceDict(model, heads + layers) as ret:
            model(**enc)
            for level, names in (("head", heads), ("layer", layers)):
                acts = torch.stack([
                    (ret[n].output[0] if isinstance(ret[n].output, tuple) else ret[n].output).float()
                    for n in names], dim=1)                          # (B, L, T, D)
                mean = (acts * mask).sum(2) / mask.sum(2)            # (B, L, D)
                out[f"{level}_mean"].append(mean.cpu().numpy())
                out[f"{level}_last"].append(acts[rows, :, last].cpu().numpy())

    out = {k: np.concatenate(v) for k, v in out.items()}
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise FloatingPointError(f"Non-finite activations in {k}")
    return out


def score_and_direction(X, y):
    """X: (N, L, H, D), y in {+1, -1}. Returns scores (L, H) and directions (L, H, D)."""
    X = X.astype(np.float64)
    pos, neg = X[y == 1], X[y == -1]
    mean_pos, mean_neg, mean_all = pos.mean(0), neg.mean(0), X.mean(0)
    between = ((mean_pos - mean_all) ** 2 + (mean_neg - mean_all) ** 2).mean(-1) / 2
    within = (pos.var(0).mean(-1) + neg.var(0).mean(-1)) / 2
    return (between / (within + 1e-8)).astype(np.float32), (mean_pos - mean_neg).astype(np.float32)


def process_chunk(model, tokenizer, axes, keys, batch_size, n_heads):
    sentences = [s for k in keys for s, _ in axes[k]["examples"]]
    labels = [np.array([l for _, l in axes[k]["examples"]]) for k in keys]
    acts = extract(model, tokenizer, sentences, batch_size)

    result = {"axis_keys": np.array(keys)}
    for name in OUTPUTS:
        A = acts[name]
        A = A.reshape(A.shape[0], A.shape[1], n_heads, -1) if name.startswith("head") \
            else A[:, :, None, :]                                    # layers: one "head"
        scores, dirs, start = [], [], 0
        for y in labels:
            s, d = score_and_direction(A[start:start + len(y)], y)
            scores.append(s)
            dirs.append(d)
            start += len(y)
        result[f"{name}_scores"] = np.stack(scores)
        result[f"{name}_directions"] = np.stack(dirs)
    return result


def save_npz(path, **arrays):
    """Write to a temporary file first, so a disconnect never leaves a broken file."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Head- and layer-level directions for all semantic axes.")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data", required=True, help="probing dataset from 1-data.py")
    ap.add_argument("--out-dir", required=True, help="checkpoint folder (e.g. on Google Drive)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=20, help="axes per checkpoint")
    ap.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo, e.g. user/axeslens")
    ap.add_argument("--keep-local", action="store_true", help="keep merged files after upload")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        probing = json.load(f)
    axes = probing["axes"]
    config = f"{probing['template_type']}_n{probing['n']}"
    model_short = args.model.split("/")[-1]
    run = f"{model_short}_{config}"
    chunk_dir = os.path.join(args.out_dir, run + "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    tokenizer = pu.load_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = pu.load_model(args.model)                                # bfloat16
    n_heads = model.config.num_attention_heads

    # ---- chunks with checkpoints ----
    keys = list(axes)
    chunks = [keys[i:i + args.chunk_size] for i in range(0, len(keys), args.chunk_size)]
    for c, chunk_keys in enumerate(tqdm(chunks, desc=run)):
        path = os.path.join(chunk_dir, f"chunk_{c:04d}.npz")
        if os.path.exists(path):
            continue                                                 # already done
        save_npz(path, **process_chunk(model, tokenizer, axes, chunk_keys,
                                       args.batch_size, n_heads))

    # ---- merge: one file per output ----
    paths = [os.path.join(chunk_dir, f"chunk_{c:04d}.npz") for c in range(len(chunks))]
    merged = []
    for name in OUTPUTS:
        parts = [np.load(p) for p in paths]
        out_path = os.path.join(args.out_dir, f"{run}_{name}.npz")
        save_npz(out_path,
                 axis_keys=np.concatenate([p["axis_keys"] for p in parts]),
                 scores=np.concatenate([p[f"{name}_scores"] for p in parts]),
                 directions=np.concatenate([p[f"{name}_directions"] for p in parts]))
        merged.append((name, out_path))
        print(f"Saved {out_path}")
    shutil.rmtree(chunk_dir)

    # ---- upload to Hugging Face ----
    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        for name, path in merged:
            api.upload_file(path_or_fileobj=path, repo_id=args.hf_repo, repo_type="dataset",
                            path_in_repo=f"{model_short}/{config}/{name}.npz")
            print(f"Uploaded {model_short}/{config}/{name}.npz")
            if not args.keep_local:
                os.remove(path)


if __name__ == "__main__":
    main()
