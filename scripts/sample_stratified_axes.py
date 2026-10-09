#!/usr/bin/env python3
"""
sample_stratified_axes.py
=========================
Builds the eligible pool of WordNet semantic axes and draws the stratified
sample of 100 axes for human annotation, following Appendix B.2.1.

Steps
-----
1. Stratum assignment (BabelDomains labels of the two poles):
     both poles Unknown              -> "Unknown"
     one pole labeled                -> that pole's label
     both poles with the same label  -> that label
     poles with different labels     -> discarded
   Labels are grouped into three strata: "Philosophy and psychology",
   "Unknown", and "Technical domains" (all remaining domains).
2. Filters:
     (a) a pole word is a quantifier or determiner (more, fewer, all, ...)
     (b) a pole definition contains a parenthetical qualifier that does not
         refer to humans, e.g. "(of plants)", "(of mammals)"
     (c) a pole word has a Zipf frequency below --min-zipf (wordfreq)
3. Stratified random sample with fixed quotas per stratum (default 34/33/33).
   Pairs listed in --exclude (e.g. non-gradable pairs such as dead/alive) are
   skipped and replaced by the next pair from the same stratum.

Input : data/processed/antonym_axes.json   (scripts/build_antonym_axes.py)
Output: eligible_pool.json, stratified_sample.json

Usage
-----
  python scripts/sample_stratified_axes.py \
      --axes data/processed/antonym_axes.json \
      --annotations data/raw/stereotypicality_annotations.csv

Requirements: Python 3.8+, wordfreq.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

from wordfreq import zipf_frequency

PSY = "Philosophy and psychology"
UNK = "Unknown"
TECH = "Technical domains"
STRATA = [PSY, UNK, TECH]

# (a) quantifiers / determiners that are not attributes
QUANTIFIERS = {
    "some", "no", "all", "more", "fewer", "less", "many", "few", "much",
    "any", "every", "each", "both", "neither", "either",
}

# (b) parenthetical qualifiers that restrict a sense to non-humans
NON_HUMAN_QUALIFIER = re.compile(
    r"\((?:of |used of |especially of )"
    r"(?:plants?|ferns?|mammals?|animals?|fungi|organisms?|texts?|linens?|"
    r"clothes?|fabrics?|parasites?|rust fungi|organic compounds?|"
    r"chemical reactions?|compounds?|muscles?|trees?|flowers?|birds?|fish|"
    r"insects?|reptiles?|amphibians?|horses?|dogs?|cats?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_axis(key, text):
    """Parse one entry of antonym_axes.json."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    field = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in lines}
    syn_neg, syn_pos = field["Antonym Pair"].split("|")
    def_neg = field["Negative"].split("\u2192", 1)[1].strip()
    def_pos = field["Positive"].split("\u2192", 1)[1].strip()
    dom_neg, dom_pos = [d.strip() for d in field["Domain"].split(" - ", 1)]
    return {
        "key": key,
        "neg": syn_neg, "pos": syn_pos,
        "neg_word": syn_neg.split(".")[0], "pos_word": syn_pos.split(".")[0],
        "neg_def": def_neg, "pos_def": def_pos,
        "neg_domain": dom_neg, "pos_domain": dom_pos,
        "text": text,
    }


def domain_of(axis):
    """Single domain label of an axis, or None if the poles disagree."""
    a, b = axis["neg_domain"], axis["pos_domain"]
    if a == UNK and b == UNK:
        return UNK
    if a == UNK:
        return b
    if b == UNK or a == b:
        return a
    return None


def stratum_of(domain):
    if domain in (PSY, UNK):
        return domain
    return TECH


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def zipf(word):
    return zipf_frequency(word.replace("_", " "), "en")


def exclusion_reason(axis, min_zipf):
    words = (axis["neg_word"], axis["pos_word"])
    if any(w.lower() in QUANTIFIERS for w in words):
        return "quantifier"
    if any(NON_HUMAN_QUALIFIER.search(d) for d in (axis["neg_def"], axis["pos_def"])):
        return "non_human_qualifier"
    if min(zipf(w) for w in words) < min_zipf:
        return "low_frequency"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stratified sample of semantic axes.")
    ap.add_argument("--axes", default="data/processed/antonym_axes.json")
    ap.add_argument("--min-zipf", type=float, default=3.0)
    ap.add_argument("--quotas", type=int, nargs=3, default=[34, 33, 33],
                    metavar=("PSY", "UNK", "TECH"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="axis keys to skip and replace (non-gradable pairs)")
    ap.add_argument("--pool-out", default="eligible_pool.json")
    ap.add_argument("--sample-out", default="stratified_sample.json")
    ap.add_argument("--annotations", default=None,
                    help="optional: annotation CSV to compare the sample with")
    args = ap.parse_args()

    with open(args.axes, encoding="utf-8") as f:
        raw = json.load(f)
    axes = [parse_axis(k, v) for k, v in raw.items()]

    # --- step 1: strata before filtering
    before = Counter()
    pool, reasons = [], Counter()
    for ax in axes:
        dom = domain_of(ax)
        # axes with conflicting labels are counted in the technical stratum
        # before filtering (they are not Unknown or Psychology on both poles)
        before[stratum_of(dom) if dom else TECH] += 1
        if dom is None:
            reasons["different_domains"] += 1
            continue
        reason = exclusion_reason(ax, args.min_zipf)
        if reason:
            reasons[reason] += 1
            continue
        ax["stratum"] = stratum_of(dom)
        pool.append(ax)

    after = Counter(ax["stratum"] for ax in pool)
    print(f"Loaded {len(axes)} axes")
    print("Excluded:", dict(reasons))
    print(f"{'Stratum':<28}{'Before':>8}{'After':>8}")
    for s in STRATA:
        print(f"{s:<28}{before[s]:>8}{after[s]:>8}")
    print(f"{'Total':<28}{sum(before.values()):>8}{len(pool):>8}")

    with open(args.pool_out, "w", encoding="utf-8") as f:
        json.dump({ax["key"]: ax["text"] for ax in pool}, f, ensure_ascii=False, indent=2)

    # --- step 3: stratified sample with fixed quotas
    rng = random.Random(args.seed)
    by_stratum = defaultdict(list)
    for ax in pool:
        by_stratum[ax["stratum"]].append(ax)
    sample = []
    for s, quota in zip(STRATA, args.quotas):
        candidates = sorted(by_stratum[s], key=lambda a: a["key"])
        rng.shuffle(candidates)
        chosen = [a for a in candidates if a["key"] not in set(args.exclude)][:quota]
        if len(chosen) < quota:
            raise ValueError(f"Stratum {s}: only {len(chosen)} eligible axes for quota {quota}")
        sample.extend(chosen)
    print(f"\nSampled {len(sample)} axes "
          f"({', '.join(f'{s}: {q}' for s, q in zip(STRATA, args.quotas))})")

    with open(args.sample_out, "w", encoding="utf-8") as f:
        json.dump({ax["key"]: ax["text"] for ax in sample}, f, ensure_ascii=False, indent=2)

    # --- optional check against the annotated axes
    if args.annotations:
        with open(args.annotations, encoding="utf-8") as f:
            annotated = {row["axis_id"] for row in csv.DictReader(f)}
        pool_keys = {ax["key"] for ax in pool}
        sample_keys = {ax["key"] for ax in sample}
        print(f"\nAnnotated axes: {len(annotated)} | in eligible pool: "
              f"{len(annotated & pool_keys)} | in this sample: {len(annotated & sample_keys)}")
        for key in sorted(annotated - pool_keys):
            ax = next(a for a in axes if a["key"] == key)
            print(f"  not in pool: {ax['neg']}|{ax['pos']} "
                  f"(reason: {exclusion_reason(ax, args.min_zipf) or 'different_domains'})")


if __name__ == "__main__":
    main()
