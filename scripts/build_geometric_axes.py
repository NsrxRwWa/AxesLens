"""Geometric axes for every semantic axis, per head and per layer, for stock Hugging Face
"""
import argparse
import json
import os
import shutil

import numpy as np
from tqdm.auto import tqdm

import hf_model_utils as hu

POSITIONS = ("mean", "last")


def score_and_direction(X, y):
    """X: (N, L, H, D), y in {+1, -1}. Returns scores (L, H) and directions (L, H, D)."""
    X = X.astype(np.float64)
    pos, neg = X[y == 1], X[y == -1]
    mean_pos, mean_neg, mean_all = pos.mean(0), neg.mean(0), X.mean(0)
    between = ((mean_pos - mean_all) ** 2 + (mean_neg - mean_all) ** 2).mean(-1) / 2
    within = (pos.var(0).mean(-1) + neg.var(0).mean(-1)) / 2
    return (between / (within + 1e-8)).astype(np.float32), (mean_pos - mean_neg).astype(np.float32)


def extract(model, tokenizer, pool, sentences, batch_size):
    device = hu.input_device(model)
    parts = {}
    for start in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[start:start + batch_size], return_tensors="pt",
                        padding=True).to(device)
        for k, v in pool(model, enc).items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def process_chunk(model, tokenizer, pool, meta, axes, keys, batch_size):
    sentences = [s for k in keys for s, _ in axes[k]["examples"]]
    labels = [np.array([l for _, l in axes[k]["examples"]]) for k in keys]
    acts = extract(model, tokenizer, pool, sentences, batch_size)
    result = {"axis_keys": np.array(keys)}
    for name, A in acts.items():
        level = name.rsplit("_", 1)[0]
        A = A.reshape(A.shape[0], A.shape[1], meta[level]["n_heads"], -1)   # (N, L, H, D)
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
    ap = argparse.ArgumentParser(description="Head- and layer-level geometric axes (Hugging Face models).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", required=True, help="probing dataset from build_probing_data.py")
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
    model_short = args.model.rstrip("/").split("/")[-1]
    run = f"{model_short}_{config}"
    chunk_dir = os.path.join(args.out_dir, run + "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    tokenizer = hu.load_tokenizer(args.model)
    model = hu.load_model(args.model)                                 # bfloat16
    points = hu.tracepoints(model)
    meta = hu.describe(points)
    for lv, m in meta.items():
        print(f"  {lv:8s} {len(m['layers']):3d} layers x {m['n_heads']:3d} heads x {m['head_dim']} dims")
    pool = hu.PooledActivations(model, points, POSITIONS)
    outputs = [f"{lv}_{pos}" for lv in meta for pos in POSITIONS]

    # ---- chunks with checkpoints ----
    keys = list(axes)
    chunks = [keys[i:i + args.chunk_size] for i in range(0, len(keys), args.chunk_size)]
    for c, chunk_keys in enumerate(tqdm(chunks, desc=run)):
        path = os.path.join(chunk_dir, f"chunk_{c:04d}.npz")
        if os.path.exists(path):
            continue                                                 # already done
        save_npz(path, **process_chunk(model, tokenizer, pool, meta, axes, chunk_keys,
                                       args.batch_size))

    # ---- merge: one file per output ----
    paths = [os.path.join(chunk_dir, f"chunk_{c:04d}.npz") for c in range(len(chunks))]
    merged = []
    for name in outputs:
        parts = [np.load(p) for p in paths]
        out_path = os.path.join(args.out_dir, f"{run}_{name}.npz")
        save_npz(out_path,
                 axis_keys=np.concatenate([p["axis_keys"] for p in parts]),
                 scores=np.concatenate([p[f"{name}_scores"] for p in parts]),
                 directions=np.concatenate([p[f"{name}_directions"] for p in parts]),
                 model_layers=np.array(meta[name.rsplit("_", 1)[0]]["layers"]))
        merged.append((name, out_path))
        print(f"Saved {out_path}")
    with open(os.path.join(args.out_dir, f"{model_short}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
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
