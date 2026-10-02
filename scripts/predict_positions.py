"""Position prediction of social groups on the Warmth and Competence semantic axes (RQ1).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PROMPTS = [                                  
    "Describe the following: {c}",
    "Provide the best description of the following using adjectives:\n{c}",
    "Concept: {c}\nWhat characteristics are associated with this concept?",
    "Consider the following: {c}\nGive a brief description.",
    "{c}\nHow would you describe the above in a few words?",
]

MODELS = {"Meta-Llama-3-8B-Instruct": {"hf": "NousResearch/Meta-Llama-3-8B-Instruct", "backend": "hf"},
          "Mistral-7B-Instruct-v0.1": {"hf": "mistralai/Mistral-7B-Instruct-v0.1", "backend": "hf"},
          "Qwen3-8B": {"hf": "Qwen/Qwen3-8B", "backend": "hf"}}
DEFAULT_MODELS = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.1"]
CONFIGS = ["listing_n30", "listing_n15", "simple_n30", "simple_n15"]
POSITIONS = ["mean"]
HEAD_KS = [1, 8, 16, 32, 64, 128, 256, 512, 1024]          # Llama / Mistral: 1024 heads
LAYER_KS = [1, 2, 4, 8, 16, 32]                            # Llama / Mistral: 32 layers
K_VALUES = {"head": HEAD_KS, "linhead": HEAD_KS, "allhead": HEAD_KS, "layer": LAYER_KS}
# Models with more heads / layers get their own lists, ending at their full size:
MODEL_K_VALUES = {
    "Qwen3-8B": {"head": HEAD_KS + [1152],                 # Qwen3-8B: 36 layers x 32 heads = 1152
                 "layer": LAYER_KS + [36]},                # Qwen3-8B: 36 layers
}


def ks_for(short, level):
    """k values for one model and level."""
    return MODEL_K_VALUES.get(short, {}).get(level, K_VALUES[level])


# a "level" is read from one or more stored activation groups:
#   head     softmax-attention heads (all layers in Llama / Mistral; 8 of 32 layers in Qwen3.5-9B)
#   linhead  Gated DeltaNet heads (Qwen3.5 only)
#   allhead  both kinds of heads, ranked together by variance ratio
#   layer    residual stream
LEVEL_GROUPS = {"head": ["head"], "linhead": ["linhead"], "allhead": ["head", "linhead"],
                "layer": ["layer"]}
METRIC = "macro_f1"
N_PERM = 1000                                 # label permutations for the chance-level test


# ----------------------------------------------------------------------------- inputs
def load_terms(xlsx):
    """Dev + Test terms with their split and Warmth / Competence signs."""
    parts = []
    for split in ["Dev", "Test"]:
        df = pd.read_excel(xlsx, sheet_name=split)
        parts.append(pd.DataFrame({"term": df["term"].astype(str), "split": split,
                                   "Warmth": df["W sign (+1/-1)"].astype(int),
                                   "Competence": df["C sign (+1/-1)"].astype(int)}))
    return pd.concat(parts, ignore_index=True)


def load_axes(warmth_file, competence_file, base_file):
    """Warmth and Competence axes; flip=True where the low pole is the base file's Positive."""
    pair = lambda text: text.splitlines()[0].split(": ", 1)[1]
    base = json.load(open(base_file, encoding="utf-8"))
    rows = []
    for category, path in [("Warmth", warmth_file), ("Competence", competence_file)]:
        for key, text in json.load(open(path, encoding="utf-8")).items():
            rows.append({"key": key, "category": category, "axis": pair(text),
                         "flip": pair(text) != pair(base[key])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- step 1: extract
def meta_path(args, short):
    return f"{args.cache_dir}/activations/{short}_meta.json"


def read_meta(args, short):
    """{group: {"n_heads": H, ...}} of the stored activations. Caches written before this
    file existed (Llama / Mistral) have 32 heads per layer and one 'head' per layer."""
    try:
        with open(meta_path(args, short)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"head": {"n_heads": 32}, "layer": {"n_heads": 1}}


def levels_of(args, short):
    groups = read_meta(args, short)
    return [lv for lv, g in LEVEL_GROUPS.items() if all(x in groups for x in g)]


def check_terms(args, short, n_terms):
    """Stop early if the cached activations were made from a different term list."""
    p = f"{args.cache_dir}/activations/{short}_layer_{POSITIONS[0]}.npy"
    if os.path.exists(p):
        n = np.load(p, mmap_mode="r").shape[0]
        if n != n_terms:
            raise SystemExit(f"{p} has {n} terms but {args.wcst} has {n_terms}: the cached "
                             f"activations are from another dataset. Move "
                             f"{args.cache_dir}/activations away and run 'extract' again.")


def extract(args):
    terms = load_terms(args.wcst)["term"].tolist()
    items = [(t, p, PROMPTS[p].format(c=term)) for t, term in enumerate(terms)
             for p in range(len(PROMPTS))]
    os.makedirs(f"{args.cache_dir}/activations", exist_ok=True)

    for short in args.models:
        name, backend = MODELS[short]["hf"], MODELS[short]["backend"]
        done = os.path.exists(meta_path(args, short)) or backend == "custom_llama"
        groups = list(read_meta(args, short)) if done else ["head", "layer"]
        paths = {f"{lv}_{pos}": f"{args.cache_dir}/activations/{short}_{lv}_{pos}.npy"
                 for lv in groups for pos in POSITIONS}
        if done and all(os.path.exists(p) for p in paths.values()):
            check_terms(args, short, len(terms))
            print(f"{short}: activations exist, skipped")
            continue
        (extract_custom_llama if backend == "custom_llama" else extract_hf)(args, short, name, terms, items)


def extract_hf(args, short, name, terms, items):
    """Stock Hugging Face model (Qwen3.5): heads via hooks on the output projections."""
    import torch
    from tqdm.auto import tqdm
    import hf_model_utils_qwen as hu

    tok = hu.load_tokenizer(name)
    model = hu.load_model(name)                     # bfloat16, as for the geometric axes
    points = hu.tracepoints(model)
    meta = hu.describe(points)
    pool = hu.PooledActivations(model, points, POSITIONS)
    paths = {f"{lv}_{pos}": f"{args.cache_dir}/activations/{short}_{lv}_{pos}.npy"
             for lv in meta for pos in POSITIONS}
    def shape(k):
        g = meta[k.rsplit("_", 1)[0]]
        return len(terms), len(PROMPTS), len(g["layers"]), g["n_heads"] * g["head_dim"]
    out = {k: np.lib.format.open_memmap(p + ".tmp", mode="w+", dtype=np.float32, shape=shape(k))
           for k, p in paths.items()}
    device = hu.input_device(model)
    for s in tqdm(range(0, len(items), args.batch_size), desc=short):
        batch = items[s:s + args.batch_size]
        enc = tok([x[2] for x in batch], return_tensors="pt", padding=True).to(device)
        acts = pool(model, enc)
        for k in paths:
            for b, (t, p, _) in enumerate(batch):
                out[k][t, p] = acts[k][b]
    for k, p in paths.items():
        out[k].flush()
        del out[k]
        os.replace(p + ".tmp", p)
    with open(meta_path(args, short), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{short}: saved activations to {args.cache_dir}/activations/")
    pool.close()
    del model
    torch.cuda.empty_cache()


def extract_custom_llama(args, short, name, terms, items):

    import torch
    from baukit import TraceDict
    from tqdm.auto import tqdm
    import pipeline_utils as pu

    paths = {f"{lv}_{pos}": f"{args.cache_dir}/activations/{short}_{lv}_{pos}.npy"
             for lv in ["head", "layer"] for pos in POSITIONS}
    tok = pu.load_tokenizer(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"                  # real tokens never see the padding
    model = pu.load_model(name)                 # bfloat16, as for the geometric axes
    L = model.config.num_hidden_layers
    names = {"head": [f"model.layers.{i}.self_attn.head_out" for i in range(L)],
             "layer": [f"model.layers.{i}" for i in range(L)]}
    dim = model.config.hidden_size
    out = {k: np.lib.format.open_memmap(p + ".tmp", mode="w+", dtype=np.float32,
                                        shape=(len(terms), len(PROMPTS), L, dim))
           for k, p in paths.items()}
    device = model.get_input_embeddings().weight.device

    for s in tqdm(range(0, len(items), args.batch_size), desc=short):
        batch = items[s:s + args.batch_size]
        enc = tok([x[2] for x in batch], return_tensors="pt", padding=True).to(device)
        mask = enc["attention_mask"][:, None, :, None].float()
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(len(batch), device=device)
        with torch.no_grad(), TraceDict(model, names["head"] + names["layer"]) as ret:
            model(**enc)
            for lv in ["head", "layer"]:
                acts = torch.stack([(ret[n].output[0] if isinstance(ret[n].output, tuple)
                                     else ret[n].output).float() for n in names[lv]], dim=1)
                res = {"mean": (acts * mask).sum(2) / mask.sum(2),
                       "last": acts[rows, :, last]}
                for pos in POSITIONS:
                    a = res[pos].cpu().numpy()
                    if not np.isfinite(a).all():
                        raise FloatingPointError(f"Non-finite activations ({lv}, {pos})")
                    for b, (t, p, _) in enumerate(batch):
                        out[f"{lv}_{pos}"][t, p] = a[b]
    for k, p in paths.items():
        out[k].flush()
        del out[k]
        os.replace(p + ".tmp", p)
    heads = model.config.num_attention_heads
    with open(meta_path(args, short), "w") as f:
        json.dump({"head": {"layers": list(range(L)), "n_heads": heads, "head_dim": dim // heads},
                   "layer": {"layers": list(range(L)), "n_heads": 1, "head_dim": dim}}, f, indent=1)
    print(f"{short}: saved activations to {args.cache_dir}/activations/")
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- geometric axes
def get_group_axes(args, short, config, group, pos, axes):
    """Scores and geometric axes of the Warmth + Competence axes, oriented low -> high.
    Read from --axes-dir if build_geometric_axes(_hf).py wrote them there, else from --repo."""
    cache = f"{args.cache_dir}/axes/{short}_{config}_{group}_{pos}.npz"
    if not os.path.exists(cache):
        os.makedirs(f"{args.cache_dir}/axes", exist_ok=True)
        local = f"{args.axes_dir}/{short}_{config}_{group}_{pos}.npz" if args.axes_dir else None
        if local and os.path.exists(local):
            path = local
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(args.repo, f"{short}/{config}/{group}_{pos}.npz", repo_type="dataset")
        d = np.load(path)
        idx = [list(d["axis_keys"]).index(k) for k in axes["key"]]
        dirs = d["directions"][idx]
        dirs[axes["flip"].to_numpy()] *= -1                 # point from low to high pole
        np.savez(cache, scores=d["scores"][idx], directions=dirs)
        del d
        if path != local:
            os.remove(os.path.realpath(path))               # free the disk
    d = np.load(cache)
    return d["scores"], d["directions"]


def get_axes(args, short, config, level, pos, axes):
    """[(scores, directions)] for each activation group of the level."""
    return [get_group_axes(args, short, config, g, pos, axes) for g in LEVEL_GROUPS[level]]


def load_acts(args, short, level, pos, rows, prompt=None):
    """[(N, L, H, D)] activations of the given terms for each group of the level,
    averaged over prompts (or one prompt)."""
    meta = read_meta(args, short)
    out = []
    for g in LEVEL_GROUPS[level]:
        a = np.load(f"{args.cache_dir}/activations/{short}_{g}_{pos}.npy", mmap_mode="r")[rows]
        a = a.mean(1) if prompt is None else a[:, prompt]
        n, L, dim = a.shape
        h = meta[g]["n_heads"]
        out.append(a.reshape(n, L, h, dim // h).astype(np.float64))
    return out


# ----------------------------------------------------------------------------- evaluation
def project_axis(groups, ks):
    """Top-k projections of all concepts on one semantic axis, for each k -> {k: (N,)}.
    groups: [(A (N, L, H, D), theta (L, H, D), score (L, H))]; with several groups (allhead),
    all their heads are ranked together by variance ratio."""
    zs, ss = [], []
    for A, theta, score in groups:
        theta = theta.astype(np.float64)
        norm2 = (theta ** 2).sum(-1)
        usable = np.isfinite(score) & np.isfinite(theta).all(-1) & (norm2 > 0)
        proj = np.einsum("nlhd,lhd->nlh", A, theta) / np.where(usable, norm2, 1.0)
        mu, sd = proj.mean(0), proj.std(0)
        z = np.where(sd > 1e-6 * np.maximum(1.0, np.abs(mu)), (proj - mu) / np.where(sd > 0, sd, 1), 0.0)
        zs.append(z.reshape(len(A), -1))
        ss.append(np.where(usable, score, -np.inf).ravel())
    z, s = np.concatenate(zs, 1), np.concatenate(ss)
    order = np.argsort(s)[::-1][:int(np.isfinite(s).sum())]
    cum = np.cumsum(z[:, order], axis=1)
    return {k: cum[:, min(k, len(order)) - 1] / min(k, len(order)) for k in ks}


def macro_f1(p, y):
    """Macro-F1 of the sign prediction (p > 0 -> +1) against labels y in {-1, +1}.
    p may be projections or +-1 predictions. Symmetric in prediction and reference."""
    pred = np.where(np.asarray(p) > 0, 1, -1)
    y = np.asarray(y)
    f1s = []
    for c in (-1, 1):
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def majority_macro_f1(y):
    """Macro-F1 of always predicting the majority label."""
    y = np.asarray(y)
    return macro_f1(np.full(len(y), 1 if (y == 1).mean() >= 0.5 else -1), y)


def baselines(terms):
    """Class balance and majority-class macro-F1 per dimension."""
    return pd.DataFrame([{"category": c, "share_positive": float((terms[c] == 1).mean()),
                          "majority_macro_f1": majority_macro_f1(terms[c].to_numpy())}
                         for c in ["Warmth", "Competence"]]).round(4)


def evaluate(A, ax_groups, axes, terms, ks):
    """Per-axis macro-F1 for every k, against the axis category's labels.
    A: [(N, L, H, D)] per group; ax_groups: [(scores, directions)] per group."""
    out = []
    for j, ax in enumerate(axes.itertuples()):
        y = terms[ax.category].to_numpy()
        groups = [(a, dirs[j], scores[j]) for a, (scores, dirs) in zip(A, ax_groups)]
        for k, p in project_axis(groups, ks).items():
            out.append({"key": ax.key, "axis": ax.axis, "category": ax.category,
                        "k": k, METRIC: macro_f1(p, y)})
    return out


def head_level(args, short):
    """Which head level a model uses for the main results (--head-level MODEL=LEVEL)."""
    return args.head_level_map.get(short, "head")


# ----------------------------------------------------------------------------- step 2: select
def select(args, terms, axes):
    dev = terms[terms.split == args.split]
    rows = []
    for short in args.models:
        for level in args.levels:
            if level not in levels_of(args, short):
                print(f"  {short}: no '{level}' activations, skipped")
                continue
            for pos in POSITIONS:
                A = load_acts(args, short, level, pos, dev.index.to_numpy())
                for config in CONFIGS:
                    ax_groups = get_axes(args, short, config, level, pos, axes)
                    for r in evaluate(A, ax_groups, axes, dev, ks_for(short, level)):
                        rows.append({"model": short, "level": level, "config": config,
                                     "position": pos, **r})
                    print(f"  {short} {level} {config} {pos} done")
    df = pd.DataFrame(rows)
    summary = (df.groupby(["model", "level", "config", "position", "category", "k"])[METRIC]
               .mean().reset_index())
    overview = summary.pivot_table(index=["level", "config", "position", "k"],
                                   columns=["model", "category"], values=METRIC).round(3)
    both = (summary.groupby(["model", "level", "config", "position", "k"])[METRIC].mean()
            .rename("mean_warmth_competence").reset_index())
    best = (both.sort_values("mean_warmth_competence", ascending=False)
            .groupby(["model", "level"]).head(5).sort_values(["level", "model"]).round(3))
    base = baselines(dev)

    path = f"{args.out_dir}/select_{args.split.lower()}.xlsx"
    with pd.ExcelWriter(path) as xl:
        overview.to_excel(xl, sheet_name="overview")
        best.to_excel(xl, sheet_name="best", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
        df.to_excel(xl, sheet_name="per_axis", index=False)
        base.to_excel(xl, sheet_name="baselines", index=False)
    print(f"Saved {path}")
    print(f"\nClass balance and majority baseline on {args.split}:\n{base.to_string(index=False)}")
    print(f"\nBest settings on {args.split} (mean of Warmth and Competence macro-F1):")
    print(best.to_string(index=False))
    plot_selection_from_excel(path, args.split)


def read_baselines(path):
    try:
        b = pd.read_excel(path, sheet_name="baselines")
        return dict(zip(b.category, b.majority_macro_f1))
    except ValueError:                                     # older file without the sheet
        return {}


def plot_selection_from_excel(path, split):
    summary = pd.read_excel(path, sheet_name="summary")
    base = read_baselines(path)
    for level in summary.level.unique():
        plot_selection(summary[summary.level == level], level, split, base,
                       path.replace(".xlsx", f"_{level}.png"))


def plot_selection(summary, level, split, base, path):
    import matplotlib.pyplot as plt
    colors = {"listing_n30": "#1f3a68", "listing_n15": "#6f94d6",
              "simple_n30": "#b5532f", "simple_n15": "#e6ac3a"}
    models = list(summary.model.unique())
    fig, axs = plt.subplots(2, len(models), figsize=(5.5 * len(models), 7.5), squeeze=False)
    for r, cat in enumerate(["Warmth", "Competence"]):
        for c, model in enumerate(models):
            ax = axs[r, c]
            ks = sorted(summary[summary.model == model]["k"].unique())   # this model's k values
            for config in CONFIGS:
                for pos in POSITIONS:
                    s = summary[(summary.model == model) & (summary.category == cat)
                                & (summary.config == config) & (summary.position == pos)].sort_values("k")
                    ax.plot([ks.index(k) for k in s["k"]], s[METRIC], marker="o", ms=4,
                            color=colors[config],
                            label=f"{config.replace('_n', ', n=').capitalize()}")
            ax.set_xticks(range(len(ks)), [int(k) for k in ks])
            if cat in base:
                ax.axhline(base[cat], color="gray", ls=":", lw=1)
            ax.set_title(f"{cat} - {model}")
            ax.set_xlabel(f"ensemble size k ({'layers' if level == 'layer' else 'heads'})")
            ax.set_ylabel(f"macro-F1 ({split})")
            ax.grid(alpha=0.3)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------- step 3: prompts
def prompts(args, terms, axes):
    dev = terms[terms.split == args.split]
    # --prompt-level head (default) or layer; output: prompts_dev.xlsx / prompts_dev_layer.xlsx
    tag = args.split.lower() + ("" if args.prompt_level == "head" else "_layer")
    rows = []
    for short in args.models:
        level = head_level(args, short) if args.prompt_level == "head" else "layer"
        ax_groups = get_axes(args, short, args.config, level, args.position, axes)
        n_units = sum(s[0].size for s, _ in ax_groups)   # all heads / all layers of this model
        k = n_units if args.k == "all" else int(args.k)
        print(f"  {short}: {level}, {args.config}, {args.position}, k={k} of {n_units}")
        for p in list(range(len(PROMPTS))) + [None]:
            A = load_acts(args, short, level, args.position, dev.index.to_numpy(), prompt=p)
            for r in evaluate(A, ax_groups, axes, dev, [k]):
                rows.append({"model": short, "prompt": f"q{p + 1}" if p is not None else "Avg.", **r})
    df = pd.DataFrame(rows)
    labels = [f"q{i + 1}" for i in range(len(PROMPTS))] + ["Avg."]
    overview = (df.groupby(["model", "category", "prompt"])[METRIC].mean()
                .unstack()[labels].round(3))
    path = f"{args.out_dir}/prompts_{tag}.xlsx"
    with pd.ExcelWriter(path) as xl:
        overview.to_excel(xl, sheet_name="overview")
        pd.DataFrame({"level": [args.prompt_level], "config": [args.config],
                      "position": [args.position], "k": [args.k],
                      "split": [args.split]}).to_excel(xl, sheet_name="setting", index=False)
        df.to_excel(xl, sheet_name="per_axis", index=False)
        baselines(dev).to_excel(xl, sheet_name="baselines", index=False)
    print(f"Saved {path}")
    print(overview.to_string())
    plot_prompts_from_excel(path)


def plot_prompts_from_excel(path, seed=42):
    """Coloured boxplots: one box per prompt and for the average, one dot per semantic axis."""
    import matplotlib.pyplot as plt
    df = pd.read_excel(path, sheet_name="per_axis")
    base = read_baselines(path)
    labels = [f"q{i + 1}" for i in range(len(PROMPTS))] + ["Avg."]
    colors = ["#4C7FB8", "#F28E2B", "#59A14F", "#D0413E", "#9B72C0", "#8C8C8C"]
    rng = np.random.default_rng(seed)
    lo = np.floor((df[METRIC].min() - 0.02) * 10) / 10
    hi = np.ceil((df[METRIC].max() + 0.02) * 10) / 10
    models = list(df.model.unique())
    fig, axs = plt.subplots(2, len(models), figsize=(5.25 * len(models), 8), sharey=True, squeeze=False)
    for r, cat in enumerate(["Warmth", "Competence"]):
        for c, model in enumerate(models):
            ax = axs[r, c]
            data = [df[(df.model == model) & (df.category == cat) & (df.prompt == q)][METRIC]
                    for q in labels]
            bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black"))
            for box, col in zip(bp["boxes"], colors):
                box.set_facecolor(col)
            for i, d in enumerate(data, 1):
                ax.scatter(i + rng.uniform(-.18, .18, len(d)), d, s=14, c="gray",
                           ec="white", lw=.4, zorder=3)
                ax.scatter(i, d.mean(), marker="D", s=45, c="white", ec="black", zorder=4)
                ax.text(i, hi + 0.04, f"{d.mean():.3f}", ha="center", fontsize=8)
            if cat in base:
                ax.axhline(base[cat], color="gray", ls=":")
            ax.set(ylim=(lo, hi + 0.08), title=f"{cat} — {model}", xlabel="concept prompt",
                   ylabel="macro-F1 per semantic axis" if c == 0 else "")
            ax.set_xticks(range(1, 7), [f"$q_{i}$" for i in range(1, 6)] + ["Avg."])
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = path.replace(".xlsx", ".png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ----------------------------------------------------------------------------- step 4: test
def permutation_p(preds, y, observed, rng, n_perm=N_PERM):
    """One-sided p-value of the mean macro-F1 over axes against chance: the human labels are
    shuffled (the same shuffle for all axes, since labels belong to terms), which keeps the
    class balance and each axis's prediction rate."""
    null = np.array([np.mean([macro_f1(p, yp) for p in preds])
                     for yp in (rng.permutation(y) for _ in range(n_perm))])
    return (1 + np.sum(null >= observed)) / (1 + n_perm), float(null.mean())


def holm(pvals):
    """Holm-corrected p-values, in the original order."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.maximum.accumulate((m - np.arange(m)) * pvals[order]).clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def test(args, terms, axes):
    from itertools import combinations
    from scipy.stats import wilcoxon
    tst = terms[terms.split == args.split]
    tag = args.split.lower()
    rng = np.random.default_rng(42)
    base = baselines(tst)
    majority = dict(zip(base.category, base.majority_macro_f1))
    per_axis, preds = [], {}
    for level, (config, pos, k) in [("head", args.head), ("layer", args.layer)]:
        k = int(k)
        for short in args.models:
            source = head_level(args, short) if level == "head" else "layer"
            ax_groups = get_axes(args, short, config, source, pos, axes)
            A = load_acts(args, short, source, pos, tst.index.to_numpy())
            for j, ax in enumerate(axes.itertuples()):
                groups = [(a, dirs[j], scores[j]) for a, (scores, dirs) in zip(A, ax_groups)]
                p = project_axis(groups, [k])[k]
                preds[(level, short, ax.key)] = np.where(p > 0, 1, -1)
                per_axis.append({"level": level, "model": short, "source": source,
                                 "config": config, "position": pos, "k": k, "key": ax.key,
                                 "axis": ax.axis, "category": ax.category,
                                 METRIC: macro_f1(p, tst[ax.category].to_numpy())})
    df = pd.DataFrame(per_axis)

    # each method against chance (label permutation) and against the majority-class baseline
    summary = []
    for (level, model, cat), g in df.groupby(["level", "model", "category"]):
        f1 = g[METRIC].to_numpy()
        boot = rng.choice(f1, size=(10000, len(f1))).mean(1)
        y = tst[cat].to_numpy()
        p_perm, chance = permutation_p([preds[(level, model, k)] for k in g["key"]], y,
                                       f1.mean(), rng)
        summary.append({"level": level, "model": model, "source": g["source"].iloc[0],
                        "category": cat, "n_axes": len(f1), METRIC: f1.mean(),
                        "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                        "chance_macro_f1": chance, "p_vs_chance_permutation": p_perm,
                        "majority_macro_f1": majority[cat],
                        "p_vs_majority_wilcoxon":
                            wilcoxon(f1 - majority[cat], alternative="greater").pvalue})
    summary = pd.DataFrame(summary).round(4)
    print(f"\nClass balance on {args.split}:\n{base.to_string(index=False)}")
    print(f"\n{args.split} set (mean macro-F1 over axes, 95% bootstrap CI; permutation test vs. "
          f"chance; one-sided Wilcoxon vs. majority class):")
    print(summary.round(3).to_string(index=False))

    # AxesLens (head) vs. layer-level variant: paired over the same axes, Holm-corrected
    hl = []
    for (model, cat), g in df.groupby(["model", "category"]):
        w = g.pivot(index="key", columns="level", values=METRIC).dropna()
        hl.append({"model": model, "category": cat, "n_axes": len(w),
                   "head": w["head"].mean(), "layer": w["layer"].mean(),
                   "diff": (w["head"] - w["layer"]).mean(),
                   "p": wilcoxon(w["head"], w["layer"], alternative="greater").pvalue})
    hl = pd.DataFrame(hl)
    hl["p_holm"] = holm(hl["p"])
    hl = hl.round(4)
    print("\nHead vs. layer (paired one-sided Wilcoxon over axes, Holm-corrected):")
    print(hl.to_string(index=False))

    # agreement between Human and every model, and between every pair of models (head level)
    agree = []
    for cat in ["Warmth", "Competence"]:
        keys = axes[axes.category == cat]["key"]
        y = tst[cat].to_numpy()
        row = {"category": cat}
        for m in args.models:
            row[f"Human-{m}"] = np.mean([macro_f1(preds[("head", m, k)], y) for k in keys])
        for m1, m2 in combinations(args.models, 2):
            row[f"{m1}-{m2}"] = np.mean([macro_f1(preds[("head", m1, k)], preds[("head", m2, k)])
                                         for k in keys])
        agree.append(row)
    agree = pd.DataFrame(agree).round(3)
    print("\nPairwise agreement (head level, macro-F1):")
    print(agree.to_string(index=False))

    path = f"{args.out_dir}/final_{tag}.xlsx"
    with pd.ExcelWriter(path) as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        hl.to_excel(xl, sheet_name="head_vs_layer", index=False)
        agree.to_excel(xl, sheet_name="agreement", index=False)
        base.to_excel(xl, sheet_name="baselines", index=False)
        df.to_excel(xl, sheet_name="per_axis", index=False)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------- plot only
def plot_all(args):
    import glob
    for path in sorted(glob.glob(f"{args.out_dir}/select_*.xlsx")):
        plot_selection_from_excel(path, os.path.basename(path)[7:-5].capitalize())
    for path in sorted(glob.glob(f"{args.out_dir}/prompts_*.xlsx")):
        plot_prompts_from_excel(path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Position prediction on Warmth / Competence axes.")
    ap.add_argument("step", choices=["extract", "select", "prompts", "test", "plot"])
    ap.add_argument("--out-dir", default="results",
                    help="Excel files and figures (default: ./results in the current directory)")
    ap.add_argument("--cache-dir", default="/content/cache" if os.path.isdir("/content") else "cache",
                    help="large intermediate files: activations and axis cache")
    ap.add_argument("--wcst", default="data/raw/WCST-final-dataset.xlsx")
    ap.add_argument("--warmth", default="data/processed/warmth_axes.json")
    ap.add_argument("--competence", default="data/processed/competence_axes.json")
    ap.add_argument("--base", default="data/processed/antonym_axes.json")
    ap.add_argument("--repo", default="NsrxRwWa/AxesLens")
    ap.add_argument("--axes-dir", default=None,
                    help="folder with geometric axes from build_geometric_axes(_hf).py "
                         "(<model>_<config>_<level>_<position>.npz); used before --repo")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS),
                    help="models to run (default: Llama and Mistral, as in the paper; "
                         "add Qwen3-8B / Qwen3.5-9B explicitly)")
    ap.add_argument("--levels", nargs="+", default=["head", "layer"], choices=list(LEVEL_GROUPS),
                    help="select: levels to compare (linhead / allhead exist for Qwen3.5 only)")
    ap.add_argument("--head-level", nargs="*", default=[], metavar="MODEL=LEVEL",
                    help="prompts/test: head level per model, e.g. Qwen3.5-9B=allhead "
                         "(default: head)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", choices=["Dev", "Test"], default=None,
                    help="terms to evaluate (default: Dev for select/prompts, Test for test)")
    ap.add_argument("--config", default="listing_n15", help="prompts: configuration")
    ap.add_argument("--position", default="mean", help="prompts: token position")
    ap.add_argument("--k", default="128",
                    help="prompts: number of heads (or layers); 'all' = every one of that model")
    ap.add_argument("--prompt-level", choices=["head", "layer"], default="head",
                    help="prompts: run the prompt ablation on heads or on the layer baseline")
    ap.add_argument("--head", nargs=3, metavar=("CONFIG", "POSITION", "K"),
                    default=["listing_n15", "mean", "128"], help="test: head-level setting")
    ap.add_argument("--layer", nargs=3, metavar=("CONFIG", "POSITION", "K"),
                    default=["listing_n15", "mean", "4"], help="test: layer-level setting")
    args = ap.parse_args()
    args.head_level_map = dict(x.split("=", 1) for x in args.head_level)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.step == "extract":
        return extract(args)                              # always all terms, both splits
    if args.step == "plot":
        return plot_all(args)
    args.split = args.split or ("Test" if args.step == "test" else "Dev")
    print(f"Evaluating on: {args.split}")
    terms, axes = load_terms(args.wcst), load_axes(args.warmth, args.competence, args.base)
    for short in args.models:
        check_terms(args, short, len(terms))
    print(f"{len(terms)} terms ({(terms.split == 'Dev').sum()} Dev, {(terms.split == 'Test').sum()} Test), "
          f"{(axes.category == 'Warmth').sum()} Warmth + {(axes.category == 'Competence').sum()} "
          f"Competence axes ({axes.flip.sum()} flipped to low -> high)")
    {"select": select, "prompts": prompts, "test": test}[args.step](args, terms, axes)


if __name__ == "__main__":
    main()
