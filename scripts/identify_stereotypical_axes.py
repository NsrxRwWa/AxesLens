"""Stereotypicality identification of semantic axes (RQ2).

A semantic axis is flagged as stereotypical when the projections of social group mentions
onto its geometric axis differ from those of frequency-matched random phrases
(two-sample Kolmogorov-Smirnov test, two-sided).

Same pipeline as predict_positions.py:
  * concept activations: hf_model_utils hooks, the five concept prompts, mean over tokens,
    averaged over the prompts
  * geometric axes: from --axes-dir or the Hugging Face dataset (--repo)
  * projection: predict_positions.project_axis (z-scored per head over social + random
    concepts together, averaged over the top-k heads or layers ranked by variance ratio)

Steps
  extract   activations of the social groups and random phrases (GPU)
  test      KS test per semantic axis, on the 100 stratified axes (--axes stratified,
            compared with the human annotations) or on all 1,999 axes (--axes all)
  power     post-hoc power and minimum detectable effect of the KS tests (from 'test')

Usage
  python scripts/identify_stereotypical_axes.py extract --models Meta-Llama-3-8B-Instruct Mistral-7B-Instruct-v0.1 Qwen3-8B
  python scripts/identify_stereotypical_axes.py test --axes stratified --models ... --head listing_n15 mean 128
  python scripts/identify_stereotypical_axes.py test --axes all --models ... --head listing_n15 mean 128
  python scripts/identify_stereotypical_axes.py power --axes stratified --models ...
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import predict_positions as pp

SETS = ("social", "random")


# ----------------------------------------------------------------------------- inputs
def read_concepts(path):
    """One concept per line; 'QID<TAB>name' lines are accepted too (the name is used)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t")[-1].strip() for line in f if line.strip()]


def read_antonym_axes(path):
    """{key: 'neg.a.NN|pos.a.NN'} for all 1,999 semantic axes."""
    pair = lambda text: text.splitlines()[0].split(": ", 1)[1].strip()
    return {k: pair(v) for k, v in json.load(open(path, encoding="utf-8")).items()}


def read_human(path, neutral_is_stereotypical=True):
    """Per-axis human labels (data/raw/stereotypicality_annotations.csv, one row per axis).
    applicable: 'no' if most annotators chose 'Not applicable'.
    stereotypical: majority of ratings below, at, or above 3 (as in the paper);
    with neutral_is_stereotypical=False only a majority below or above 3 counts."""
    d = pd.read_csv(path, keep_default_na=False)
    d["applicable"] = d["applicable"] == "yes"
    if neutral_is_stereotypical:
        stereo = d["stereotypical"] == "yes"
    else:
        stereo = d["majority"].isin(["below", "above"])
    d["human_stereotypical"] = stereo.astype(float).where(d["applicable"])   # NaN: not applicable
    d["pair"] = d["negative_pole"] + "|" + d["positive_pole"]
    return d.rename(columns={"axis_id": "key", "social_group": "annotated_group"})[
        ["key", "pair", "annotated_group", "applicable", "human_stereotypical"]]


# ----------------------------------------------------------------------------- step 1: extract
def act_path(args, short, which, group, pos):
    return f"{args.cache_dir}/stereo/{short}_{which}_{group}_{pos}.npy"


def meta_path(args, short):
    return f"{args.cache_dir}/stereo/{short}_meta.json"


def extract(args):
    """Activations of every concept under the five concept prompts: (N, 5, L, H*D)."""
    import torch
    import hf_model_utils as hu
    os.makedirs(f"{args.cache_dir}/stereo", exist_ok=True)
    concepts = {"social": read_concepts(args.social), "random": read_concepts(args.random)}
    for short in args.models:
        if os.path.exists(meta_path(args, short)):
            print(f"{short}: activations exist, skipped")
            continue
        name = pp.MODELS[short]["hf"]
        tok, model = hu.load_tokenizer(name), hu.load_model(name)
        points = hu.tracepoints(model)
        meta = hu.describe(points)
        pool = hu.PooledActivations(model, points, pp.POSITIONS)
        device = hu.input_device(model)
        for which, words in concepts.items():
            items = [(t, p, pp.PROMPTS[p].format(c=w)) for t, w in enumerate(words)
                     for p in range(len(pp.PROMPTS))]
            out = {}
            for s in range(0, len(items), args.batch_size):
                batch = items[s:s + args.batch_size]
                enc = tok([x[2] for x in batch], return_tensors="pt", padding=True).to(device)
                acts = pool(model, enc)
                for key, a in acts.items():
                    if key not in out:
                        out[key] = np.zeros((len(words), len(pp.PROMPTS)) + a.shape[1:], np.float32)
                    for b, (t, p, _) in enumerate(batch):
                        out[key][t, p] = a[b]
            for key, a in out.items():
                group, pos = key.rsplit("_", 1)
                np.save(act_path(args, short, which, group, pos), a)
            print(f"{short}: {which} ({len(words)} concepts) done")
        with open(meta_path(args, short), "w") as f:
            json.dump(meta, f, indent=1)
        pool.close()
        del model
        torch.cuda.empty_cache()


def load_acts(args, short, which, level, pos):
    """[(N, L, H, D)] per activation group of the level, averaged over the prompts."""
    meta = json.load(open(meta_path(args, short)))
    out = []
    for g in pp.LEVEL_GROUPS[level]:
        a = np.load(act_path(args, short, which, g, pos)).mean(1)
        n, L, dim = a.shape
        h = meta[g]["n_heads"]
        out.append(a.reshape(n, L, h, dim // h).astype(np.float64))
    return out


# ----------------------------------------------------------------------------- geometric axes
def load_geometric_axes(args, short, config, group, pos, keys):
    """scores (A, L, H) and directions (A, L, H, D) of the given semantic axes."""
    local = f"{args.axes_dir}/{short}_{config}_{group}_{pos}.npz" if args.axes_dir else None
    if local and os.path.exists(local):
        path = local
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.repo, f"{short}/{config}/{group}_{pos}.npz", repo_type="dataset")
    d = np.load(path)
    index = {k: i for i, k in enumerate(d["axis_keys"])}
    idx = [index[k] for k in keys]
    return d["scores"][idx], d["directions"][idx]


# ----------------------------------------------------------------------------- step 2: test
def project_axes(A, ax, ks):
    """predict_positions.project_axis for many semantic axes at once -> {k: (axes, N)}.
    A: [(N, L, H, D)] per group; ax: [(scores (S, L, H), directions (S, L, H, D))] per group."""
    zs, ss = [], []
    for a, (score, theta) in zip(A, ax):
        theta = theta.astype(np.float64)
        norm2 = (theta ** 2).sum(-1)                                      # (S, L, H)
        usable = np.isfinite(score) & np.isfinite(theta).all(-1) & (norm2 > 0)
        proj = np.einsum("nlhd,slhd->snlh", a, np.nan_to_num(theta)) / np.where(usable, norm2, 1.0)[:, None]
        mu, sd = proj.mean(1, keepdims=True), proj.std(1, keepdims=True)
        ok = sd > 1e-6 * np.maximum(1.0, np.abs(mu))
        z = np.where(ok, (proj - mu) / np.where(sd > 0, sd, 1), 0.0)
        zs.append(z.reshape(z.shape[0], z.shape[1], -1))                 # (S, N, L*H)
        ss.append(np.where(usable, score, -np.inf).reshape(score.shape[0], -1))
    z, s = np.concatenate(zs, 2), np.concatenate(ss, 1)
    order = np.argsort(s, axis=1, kind="stable")[:, ::-1]                # best first, per axis
    n_ok = np.isfinite(s).sum(1)                                          # usable units per axis
    cum = np.cumsum(np.take_along_axis(z, order[:, None, :], axis=2), axis=2)
    out = {}
    for k in ks:
        kk = np.minimum(k, n_ok)                                          # (S,)
        out[k] = cum[np.arange(len(kk)), :, kk - 1] / kk[:, None]
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def agreement(df):
    """Model (KS) vs. human stereotypicality labels on the applicable axes."""
    def cohen_kappa(a, b):
        po = np.mean(a == b)
        pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
        return (po - pe) / (1 - pe) if pe < 1 else np.nan

    rows = []
    for (model, level, k), g in df.dropna(subset=["human_stereotypical"]).groupby(["model", "level", "k"]):
        h = g["human_stereotypical"].astype(bool).to_numpy()
        m = g["stereotypical"].astype(bool).to_numpy()
        tp, tn = int((h & m).sum()), int((~h & ~m).sum())
        fp, fn = int((~h & m).sum()), int((h & ~m).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        rows.append({"model": model, "level": level, "k": k, "n_axes": len(g),
                     "human_stereotypical": int(h.sum()), "model_stereotypical": int(m.sum()),
                     "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                     "accuracy": (tp + tn) / len(g), "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else np.nan,
                     "cohen_kappa": cohen_kappa(h, m)})
    return pd.DataFrame(rows)


def test(args):
    pairs = read_antonym_axes(args.base)
    human = read_human(args.human, not args.neutral_not_stereotypical) if args.axes == "stratified" else None
    keys = list(human["key"]) if human is not None else list(pairs)
    social_words, random_words = read_concepts(args.social), read_concepts(args.random)
    n_soc = len(social_words)
    settings = [("head", args.head)] + ([("layer", args.layer)] if args.layer else [])
    rows, projections = [], []
    for short in args.models:
        for level, (config, pos, *ks) in settings:
            ks = [int(k) for k in ks]
            A_soc = load_acts(args, short, "social", level, pos)
            A_rnd = load_acts(args, short, "random", level, pos)
            A = [np.concatenate([s, r]) for s, r in zip(A_soc, A_rnd)]   # one z-score reference
            ax = [load_geometric_axes(args, short, config, g, pos, keys) for g in pp.LEVEL_GROUPS[level]]
            # all axes at once (in batches of --axis-batch), instead of one axis after another
            for start in range(0, len(keys), args.axis_batch):
                batch = keys[start:start + args.axis_batch]
                sub = [(s[start:start + len(batch)], d[start:start + len(batch)]) for s, d in ax]
                for k, P in project_axes(A, sub, ks).items():              # P: (axes, concepts)
                    soc, rnd = P[:, :n_soc], P[:, n_soc:]
                    D, pval = ks_2samp(soc, rnd, alternative="two-sided", axis=1)
                    rows.append(pd.DataFrame({
                        "model": short, "level": level, "config": config, "position": pos, "k": k,
                        "key": batch, "pair": [pairs[x] for x in batch], "ks_D": D, "ks_p": pval,
                        "median_social": np.median(soc, 1), "median_random": np.median(rnd, 1)}))
                    projections.append(pd.DataFrame({
                        "model": short, "level": level, "k": k,
                        "key": np.repeat(batch, P.shape[1]),
                        "set": np.tile(["social"] * n_soc + ["random"] * rnd.shape[1], len(batch)),
                        "concept": np.tile(social_words + random_words, len(batch)),
                        "projection": P.ravel()}))
            print(f"  {short} {level} {config} {pos} k={ks}: {len(keys)} axes done")
    df = pd.concat(rows, ignore_index=True)
    df["ks_p_bh"] = df.groupby(["model", "level", "k"])["ks_p"].transform(bh)
    p_col = "ks_p_bh" if args.correction == "bh" else "ks_p"
    df["stereotypical"] = df[p_col] < args.alpha
    if human is not None:
        df = df.merge(human.drop(columns="pair"), on="key", how="left")

    counts = (df.groupby(["model", "level", "k"])
              .agg(n_axes=("key", "size"), n_stereotypical=("stereotypical", "sum")).reset_index())
    counts["share_stereotypical"] = counts["n_stereotypical"] / counts["n_axes"]
    tag = f"{args.axes}"
    path = f"{args.out_dir}/stereotypicality_{tag}.xlsx"
    os.makedirs(args.out_dir, exist_ok=True)
    with pd.ExcelWriter(path) as xl:
        counts.to_excel(xl, sheet_name="summary", index=False)
        if human is not None:
            agree = agreement(df).round(3)
            agree.to_excel(xl, sheet_name="agreement_with_humans", index=False)
        df.sort_values(["model", "level", "k", "ks_p"]).to_excel(xl, sheet_name="per_axis", index=False)
    pd.concat(projections).to_csv(f"{args.out_dir}/stereotypicality_{tag}_projections.csv.gz",
                                  index=False)
    print(f"\nKS test (alpha={args.alpha}, {'Benjamini-Hochberg' if args.correction == 'bh' else 'uncorrected'}):")
    print(counts.round(3).to_string(index=False))
    if human is not None:
        print(f"\nAgreement with human annotations ({int(human['applicable'].sum())} applicable axes):")
        print(agree.to_string(index=False))
    print(f"Saved {path}")
    if args.plot_axes:
        plot_axes(df, pd.concat(projections), args, tag)
    if human is not None:
        for (model, level, k), g in df.dropna(subset=["human_stereotypical"]).groupby(["model", "level", "k"]):
            plot_agreement(g, f"{args.out_dir}/plots/agreement_{tag}_{model}_{level}_k{k}.png",
                           f"{model}, {level}, k={k}", args.plot_axes)


# ----------------------------------------------------------------------------- plots
def same_axis(pair):
    """Matcher for an axis name in either pole order ('a|b' or 'b|a')."""
    want = frozenset(s.strip() for s in pair.split("|"))
    return lambda p: frozenset(s.strip() for s in str(p).split("|")) == want


def plot_axes(df, proj, args, tag):
    """Strip plot of social vs. random projections for the given semantic axes."""
    import matplotlib.pyplot as plt
    os.makedirs(f"{args.out_dir}/plots", exist_ok=True)
    for pair in args.plot_axes:
        sel = df[df["pair"].map(same_axis(pair))]
        if sel.empty:
            print(f"  [plot] {pair!r} not among the tested axes, skipped")
            continue
        for r in sel.itertuples():
            p = proj[(proj.model == r.model) & (proj.level == r.level) & (proj.k == r.k)
                     & (proj.key == r.key)]
            fig, ax = plt.subplots(figsize=(6, 3.2))
            rng = np.random.default_rng(0)
            for y, (which, color) in enumerate([("social", "crimson"), ("random", "steelblue")]):
                v = p[p.set == which]
                ax.scatter(v.projection, y + rng.normal(0, 0.05, len(v)), s=30, color=color, alpha=0.7)
                ax.plot([v.projection.median()] * 2, [y - 0.2, y + 0.2], color=color, lw=2)
                for i in (v.projection.idxmin(), v.projection.idxmax()):
                    ax.annotate(v.concept[i], (v.projection[i], y), xytext=(0, 9),
                                textcoords="offset points", ha="center", fontsize=8, color=color)
            ax.set_yticks([0, 1], ["Social groups", "Random"])
            ax.set_xlabel(f"projection onto {pair}")
            ax.set_title(f"{r.model}, {r.level}, k={r.k}: KS D={r.ks_D:.3f}, p={r.ks_p:.1e}", fontsize=9)
            fig.tight_layout()
            out = f"{args.out_dir}/plots/{tag}_{r.model}_{r.level}_k{r.k}_{pair.replace('|', '_vs_')}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)


def plot_agreement(g, path, title, label_axes=None):
    """Human vs. model verdict per axis (diagonal = agreement), as in the previous version."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    g = g.sort_values("pair").reset_index(drop=True)
    h, m = g["human_stereotypical"].astype(bool).to_numpy(), g["stereotypical"].to_numpy()
    cat = np.select([h & m, ~h & ~m, h & ~m], [0, 1, 2], 3)
    style = {0: ("#15803d", "o", "Both: stereotype"), 1: ("#2563eb", "s", "Both: non-stereotype"),
             2: ("#dc2626", "^", "Only humans"), 3: ("#ea580c", "X", "Only model")}
    i = np.arange(len(g))
    x = np.where(h | ~m, i, 0)              # only model: moved to the left edge
    y = np.where(h & ~m, 0, i)              # only humans: moved to the bottom edge
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.plot([-0.5, len(g) - 0.5], [-0.5, len(g) - 0.5], "--", color="0.5", lw=1)
    for c, (col, mk, _) in style.items():
        s = cat == c
        ax.scatter(x[s], y[s], c=col, marker=mk, s=55, edgecolors="white", lw=0.6, zorder=3)
    for pair in label_axes or []:
        hit = np.flatnonzero(g["pair"].map(same_axis(pair)).to_numpy())
        for j in hit:
            ax.annotate(pair.replace("|", " ↔ "), (x[j], y[j]), xytext=(8, -8),
                        textcoords="offset points", fontsize=8, style="italic")
    ax.set(xticks=[], yticks=[], xlim=(-0.6, len(g) - 0.4), ylim=(-0.6, len(g) - 0.4),
           xlabel="Human judgment", ylabel="Model prediction")
    ax.set_title(f"{title}: agreement {(h == m).sum()}/{len(g)}", fontsize=10)
    ax.set_aspect("equal")
    handles = [Line2D([0], [0], marker=mk, ls="None", color=col, ms=9,
                      label=f"{lab} (n={(cat == c).sum()})") for c, (col, mk, lab) in style.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- step 3: power
def power(args):
    """Post-hoc power at the observed sample sizes and the minimum detectable effect (MDE,
    in pooled-SD units) of each KS test, by Monte Carlo simulation (as in the previous version)."""
    proj = pd.read_csv(f"{args.out_dir}/stereotypicality_{args.axes}_projections.csv.gz")
    rng = np.random.default_rng(args.seed)
    rows = []
    for (model, level, k, key), g in proj.groupby(["model", "level", "k", "key"]):
        soc = g[g.set == "social"].projection.to_numpy()
        rnd = g[g.set == "random"].projection.to_numpy()
        pw = np.mean([ks_2samp(rng.choice(soc, len(soc)), rng.choice(rnd, len(rnd)))[1] < args.alpha
                      for _ in range(args.n_sim)])
        pooled, mde = np.concatenate([soc, rnd]), np.nan
        for s in np.linspace(0, 2, 21):
            hits = [ks_2samp(rng.choice(pooled, len(soc)),
                             rng.choice(pooled, len(rnd)) + s * pooled.std(ddof=1))[1] < args.alpha
                    for _ in range(args.n_sim // 2)]
            if np.mean(hits) >= args.target_power:
                mde = s
                break
        rows.append({"model": model, "level": level, "k": k, "key": key,
                     "power_at_observed_n": pw, "mde_pooled_sd": mde})
    df = pd.DataFrame(rows)
    path = f"{args.out_dir}/stereotypicality_{args.axes}_power.xlsx"
    df.to_excel(path, index=False)
    print(df.groupby(["model", "level", "k"])[["power_at_observed_n", "mde_pooled_sd"]]
          .median().round(3).to_string())
    print(f"Saved {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stereotypicality identification of semantic axes (KS test).")
    ap.add_argument("step", choices=["extract", "test", "power"])
    ap.add_argument("--axes", choices=["stratified", "all"], default="stratified",
                    help="stratified: the 100 annotated axes (with human comparison); all: 1,999 axes")
    ap.add_argument("--models", nargs="+", default=list(pp.MODELS), choices=list(pp.MODELS))
    ap.add_argument("--social", default="data/raw/social_groups.txt")
    ap.add_argument("--random", default="data/raw/random_phrases.txt")
    ap.add_argument("--human", default="data/raw/stereotypicality_annotations.csv")
    ap.add_argument("--base", default="data/processed/antonym_axes.json")
    ap.add_argument("--neutral-not-stereotypical", action="store_true",
                    help="count a neutral majority (rating 3) as non-stereotypical "
                         "(default: stereotypical, as in the paper)")
    ap.add_argument("--head", nargs="+", metavar="CONFIG POSITION K", default=["listing_n15", "mean", "128"],
                    help="head setting: config, position, one or more k values")
    ap.add_argument("--layer", nargs="+", metavar="CONFIG POSITION K", default=None,
                    help="optional layer baseline, e.g. listing_n15 mean 8")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--correction", choices=["none", "bh"], default="none",
                    help="multiple-testing correction across axes (bh = Benjamini-Hochberg)")
    ap.add_argument("--plot-axes", nargs="*", default=None, metavar="neg.a.NN|pos.a.NN")
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--target-power", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--axis-batch", type=int, default=64,
                    help="semantic axes processed together (lower it if memory runs out)")
    ap.add_argument("--repo", default="NsrxRwWa/AxesLens")
    ap.add_argument("--axes-dir", default=None)
    ap.add_argument("--cache-dir", default="/content/cache" if os.path.isdir("/content") else "cache")
    ap.add_argument("--out-dir", default="results/stereotypicality")
    args = ap.parse_args()
    {"extract": extract, "test": test, "power": power}[args.step](args)


if __name__ == "__main__":
    main()
