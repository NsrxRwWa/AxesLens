#!/usr/bin/env python3
"""
build_antonym_axes.py
=====================
Builds the set of WordNet 3.0 adjective antonym axes, annotated with
BabelDomains domain labels, in a direction-neutral, fully deterministic form.

Output (JSON object, one entry per axis):

  "a00038750-a00039592": "Antonym Pair: active.a.03|passive.a.01\n
     Pole_A: active → disposed to take action or effectuate change\n
     Pole_B: passive, inactive → lacking in energy or will; - George Meredith\n
     Domain: Unknown - Unknown\n
     Confidence: Unknown - Unknown\n"

Usage:
  python build_antonym_axes.py --babel babeldomains_wordnet.txt --out antonym_axes.json
  # optional: check that the same axes / text as an older file are produced
  python build_antonym_axes.py --babel babeldomains_wordnet.txt --out antonym_axes.json \
      --reference antonym_pairs_with_domain.json

Requirements: Python 3.8+, nltk (WordNet 3.0 corpus). No other dependencies.
"""

import argparse
import json
import re
import sys

import nltk

EXPECTED_WORDNET_VERSION = "3.0"   # BabelDomains ids are WordNet 3.0 offsets


# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------
def load_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    version = wn.get_version()
    if version != EXPECTED_WORDNET_VERSION:
        sys.exit(f"WordNet version is {version}, expected {EXPECTED_WORDNET_VERSION}. "
                 f"The ids would not match BabelDomains.")
    return wn


def synset_id(syn):
    """WordNet 3.0 identifier, e.g. Synset('able.a.01') -> 'a00001740'."""
    return f"{syn.pos()}{syn.offset():08d}"


def collect_axes(wn):
    """Return a list of (syn_A, syn_B), unique at synset level, A = smaller offset."""
    axes = {}
    for syn in wn.all_synsets(pos=wn.ADJ):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                other = ant.synset()
                a, b = sorted((syn, other), key=lambda s: (s.offset(), s.pos()))
                axes[(synset_id(a), synset_id(b))] = (a, b)
    return [axes[k] for k in sorted(axes, key=lambda k: (int(k[0][1:]), int(k[1][1:])))]


# ---------------------------------------------------------------------------
# BabelDomains
# ---------------------------------------------------------------------------
def load_babeldomains(path):
    """
    Lines: <wn id> TAB <domain> TAB <confidence> [TAB <extra domain> ...]
    A leading '*' on the confidence is removed; extra domains are joined with ' | '.
    If an id occurs twice, the higher confidence wins (ties: first line wins).
    """
    info = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            wn_id = parts[0].strip()
            conf = re.sub(r"^\*", "", parts[2].strip())
            domain = " | ".join([parts[1].strip()] + [p.strip() for p in parts[3:] if p.strip()])
            if wn_id in info:
                try:
                    if float(conf) > float(info[wn_id][1]):
                        info[wn_id] = (domain, conf)
                except ValueError:
                    pass
            else:
                info[wn_id] = (domain, conf)
    return info


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def format_entry(a, b, babel):
    a_id, b_id = synset_id(a), synset_id(b)
    a_dom, a_conf = babel.get(a_id, ("Unknown", "Unknown"))
    b_dom, b_conf = babel.get(b_id, ("Unknown", "Unknown"))
    key = f"{a_id}-{b_id}"
    text = (
        f"Antonym Pair: {a.name()}|{b.name()}\n"
        f"  Pole_A: {', '.join(l.name() for l in a.lemmas())} → {a.definition()}\n"
        f"  Pole_B: {', '.join(l.name() for l in b.lemmas())} → {b.definition()}\n"
        f"  Domain: {a_dom} - {b_dom}\n"
        f"  Confidence: {a_conf} - {b_conf}\n"
    )
    return key, text


def compare_with_reference(result, ref_path):
    """Compare with an older file that used 'Negative:'/'Positive:' labels."""
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    norm = {k: v.replace("  Negative: ", "  Pole_A: ").replace("  Positive: ", "  Pole_B: ")
            for k, v in ref.items()}
    only_new = sorted(set(result) - set(norm))
    only_ref = sorted(set(norm) - set(result))
    diff_text = [k for k in result if k in norm and result[k] != norm[k]]
    print(f"Reference: {len(ref)} entries | new only: {len(only_new)} | "
          f"reference only: {len(only_ref)} | different text: {len(diff_text)}")
    for k in (only_new + only_ref + diff_text)[:10]:
        print("   ", k)
    return not (only_new or only_ref or diff_text)


def main():
    ap = argparse.ArgumentParser(description="Build WordNet 3.0 adjective antonym axes.")
    ap.add_argument("--babel", required=True, help="babeldomains_wordnet.txt")
    ap.add_argument("--out", default="antonym_axes.json")
    ap.add_argument("--reference", default=None, help="optional older file to compare with")
    args = ap.parse_args()

    wn = load_wordnet()
    babel = load_babeldomains(args.babel)
    axes = collect_axes(wn)

    result = {}
    for a, b in axes:
        key, text = format_entry(a, b, babel)
        assert key not in result, f"duplicate key {key}"
        result[key] = text

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(result)} axes to {args.out}")

    if args.reference and not compare_with_reference(result, args.reference):
        sys.exit("Output differs from the reference file (see above).")


if __name__ == "__main__":
    main()
