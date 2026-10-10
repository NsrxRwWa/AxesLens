#!/usr/bin/env python3
"""
build_eligible_pool.py
======================
Builds the eligible pool of WordNet semantic axes relevant to humans,
following Appendix B.2.1.

Steps
-----
1. Domain assignment (BabelDomains labels of the two poles):
     both poles Unknown              -> "Unknown"
     one pole labeled                -> that pole's label
     both poles with the same label  -> that label
     poles with different labels     -> discarded
   Labels are grouped into three strata: "Philosophy and psychology",
   "Unknown", and "Technical domains" (all remaining domains); the strata
   are only reported, not sampled.
2. Filters:
     (a) a pole word is a quantifier or determiner (more, fewer, all, ...)
     (b) a pole definition contains a parenthetical qualifier that does not
         refer to humans, e.g. "(of plants)", "(of mammals)"
     (c) a pole word has a Zipf frequency below --min-zipf (wordfreq)
     (d) the axis is in NON_HUMAN_AXES: axes whose poles are restricted to
         non-human referents (weather, music, grammar, clothing, ...) that
         the qualifier pattern in (b) does not catch

Input : data/processed/antonym_axes.json   (scripts/build_antonym_axes.py)
Output: eligible_pool.json  {axis key: axis text} of all axes that pass

Usage
-----
  python scripts/build_eligible_pool.py \
      --axes data/processed/antonym_axes.json \
      --pool-out data/processed/eligible_pool.json \
      --annotations data/raw/stereotypicality_annotations.csv

Requirements: Python 3.8+, wordfreq.
"""

import argparse
import csv
import json
import re
from collections import Counter

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

# (d) axes whose poles are restricted to non-human referents (weather, music,
#     grammar, clothing, physics, ...) that the qualifier pattern in (b) does not
#     catch; key -> negative|positive synset
NON_HUMAN_AXES = {
    "a02453035-a02453183",  # a_la_carte.a.01|table_d'hote.a.01
    "a00037457-a00037757",  # active.a.01|inactive.a.02
    "a00043411-a00043615",  # active.a.09|quiet.a.06
    "a00041051-a00041361",  # active.a.12|extinct.a.02
    "a00040325-a00040685",  # active.a.13|dormant.a.02
    "a00043765-a00044353",  # actual.a.01|potential.a.01
    "a01811820-a01811905",  # acute.a.04|obtuse.a.01
    "a00676555-a00677313",  # acyclic.a.02|cyclic.a.03
    "a00050641-a00050947",  # addressed.a.01|unaddressed.a.01
    "a00333351-a00333987",  # afferent.a.01|efferent.a.01
    "a01663201-a01663359",  # alternate.a.04|opposite.a.02
    "a00763633-a00763767",  # alternating.a.01|direct.a.07
    "a00110497-a00110701",  # analogue.a.01|digital.a.03
    "a00110853-a00111129",  # analytic.a.04|synthetic.a.04
    "a02380565-a02380819",  # asynchronous.a.01|synchronous.a.02
    "a00160425-a00160573",  # attached.a.02|detached.a.04
    "a01033708-a01033840",  # backhand.a.01|forehand.a.01
    "a00945513-a00945772",  # bowed.a.01|plucked.a.01
    "a00302761-a00303727",  # calm.a.02|stormy.a.01
    "a00316572-a00316827",  # carvel-built.a.01|clinker-built.a.01
    "a00358132-a00358951",  # charged.a.01|uncharged.a.01
    "a00438166-a00438567",  # clement.a.01|inclement.a.01
    "a01659999-a01660135",  # closed.a.02|open.a.11
    "a00328528-a00328653",  # coherent.a.03|incoherent.a.02
    "a00485431-a00485593",  # commissioned.a.01|noncommissioned.a.01
    "a00597424-a00597599",  # continuous.a.02|discontinuous.a.01
    "a00610532-a00610861",  # conventional.a.03|nuclear.a.01
    "a02531919-a02532200",  # cool.a.03|warm.a.03
    "a01833643-a01833791",  # cultivated.a.01|uncultivated.a.01
    "a00662958-a00663104",  # cut.a.06|uncut.a.04
    "a00408660-a00409440",  # dark.a.02|light.a.02
    "a02537743-a02538050",  # decreasing.a.02|increasing.a.02
    "a00730215-a00730470",  # dependent.a.03|independent.a.03
    "a01147433-a01147622",  # diploid.a.01|haploid.a.01
    "a00793793-a00793988",  # dominant.a.02|recessive.a.02
    "a00794426-a00794650",  # double-breasted.a.01|single-breasted.a.01
    "a02220308-a02220571",  # double.a.04|single.a.02
    "a02367604-a02367785",  # dry.a.06|sweet.a.07
    "a00873387-a00873502",  # end-stopped.a.01|run-on.a.01
    "a01514374-a01514598",  # extensive.a.03|intensive.a.03
    "a00955626-a00955915",  # fair.a.04|foul.a.04
    "a00983573-a00983722",  # fast.a.02|slow.a.04
    "a00995119-a00995468",  # favorable.a.02|unfavorable.a.02
    "a01486084-a01486197",  # feminine.a.02|masculine.a.01
    "a01486197-a01486327",  # feminine.a.02|neuter.a.01
    "a01008439-a01008745",  # finite.a.02|infinite.a.02
    "a01577771-a01578152",  # flat.a.06|natural.a.05
    "a01577973-a01578152",  # flat.a.06|sharp.a.10
    "a00782856-a00782957",  # focused.a.01|unfocused.a.01
    "a00203774-a00203917",  # forward.a.03|reverse.a.02
    "a01456710-a01458054",  # full.a.05|thin.a.06
    "a01221502-a01221719",  # gabled.a.01|hipped.a.02
    "a01157762-a01157887",  # hard.a.07|soft.a.08
    "a01156505-a01156925",  # hard.a.08|soft.a.09
    "a01191227-a01191448",  # heavy.a.09|light.a.14
    "a01210581-a01210717",  # high-interest.a.01|low-interest.a.01
    "a01218341-a01218797",  # high-rise.a.01|low-rise.a.01
    "a01404702-a01405214",  # illegible.a.01|legible.a.01
    "a00464195-a00464399",  # inshore.a.01|offshore.a.01
    "a02402559-a02403030",  # intemperate.a.01|temperate.a.01
    "a01961937-a01962107",  # irregular.a.04|regular.a.09
    "a02293856-a02294263",  # legato.a.01|staccato.a.01
    "a01424455-a01424868",  # loaded.a.02|unloaded.a.01
    "a02596222-a02596342",  # long-spurred.a.01|short-spurred.a.01
    "a01444022-a01444230",  # long.a.06|short.a.07
    "a01452593-a01454636",  # loud.a.01|soft.a.03
    "a01469390-a01469516",  # made.a.02|unmade.a.01
    "a01472098-a01472225",  # major.a.05|minor.a.04
    "a01486084-a01486327",  # masculine.a.01|neuter.a.01
    "a01432894-a01433081",  # maxi.a.01|midi.a.01
    "a01432712-a01433081",  # maxi.a.01|mini.a.01
    "a01432712-a01432894",  # midi.a.01|mini.a.01
    "a01577771-a01577973",  # natural.a.05|sharp.a.10
    "a01649876-a01650193",  # one-piece.a.01|three-piece.a.01
    "a01649876-a01650037",  # one-piece.a.01|two-piece.a.01
    "a02182862-a02182979",  # plural.a.02|singular.a.05
    "a01877617-a01877919",  # progressive.a.03|regressive.a.01
    "a02290265-a02290714",  # quantitative.a.03|syllabic.a.03
    "a02591896-a02592015",  # re-entrant.a.01|salient.a.02
    "a01240591-a01241065",  # running.a.01|standing.a.03
    "a00757236-a00757408",  # saturated.a.02|unsaturated.a.02
    "a00393508-a00393852",  # saturated.a.03|unsaturated.a.03
    "a00175528-a00175719",  # sonic.a.01|subsonic.a.01
    "a00175528-a00175887",  # sonic.a.01|supersonic.a.01
    "a02277279-a02277485",  # sparkling.a.02|still.a.05
    "a00175719-a00175887",  # subsonic.a.01|supersonic.a.01
    "a01650037-a01650193",  # three-piece.a.01|two-piece.a.01
    "a02466916-a02466999",  # tubed.a.01|tubeless.a.01
    "a02534501-a02534690",  # waning.a.01|waxing.a.01
}


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
    if axis["key"] in NON_HUMAN_AXES:
        return "non_human_listed"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Eligible pool of semantic axes relevant to humans.")
    ap.add_argument("--axes", default="data/processed/antonym_axes.json")
    ap.add_argument("--min-zipf", type=float, default=2.0)
    ap.add_argument("--pool-out", default="eligible_pool.json")
    ap.add_argument("--annotations", default=None,
                    help="optional: annotation CSV, to check that the annotated axes are in the pool")
    args = ap.parse_args()

    with open(args.axes, encoding="utf-8") as f:
        raw = json.load(f)
    axes = [parse_axis(k, v) for k, v in raw.items()]

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
    print(f"\nSaved {len(pool)} eligible axes to {args.pool_out}")

    # --- optional check: are all annotated axes in the pool?
    if args.annotations:
        with open(args.annotations, encoding="utf-8") as f:
            annotated = {row["axis_id"] for row in csv.DictReader(f)}
        pool_keys = {ax["key"] for ax in pool}
        print(f"\nAnnotated axes: {len(annotated)} | in eligible pool: "
              f"{len(annotated & pool_keys)}")
        for key in sorted(annotated - pool_keys):
            ax = next(a for a in axes if a["key"] == key)
            print(f"  not in pool: {ax['neg']}|{ax['pos']} "
                  f"(reason: {exclusion_reason(ax, args.min_zipf) or 'different_domains'})")


if __name__ == "__main__":
    main()
