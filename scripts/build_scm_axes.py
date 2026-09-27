"""Build Warmth and Competence axes from Nicolas et al. (2021) seed labels.

Retain axes with at least one matching seed synset in the category.
Infer the opposite direction for an unlabeled pole; skip conflicts.
Negative = low, Positive = high. Preserve original axis IDs and entry order.
"""
import argparse
import csv
import json
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser(
    description="Build Warmth and Competence axes from seed labels."
)
parser.add_argument("--axes", required=True, help="WordNet antonym-axis JSON")
parser.add_argument("--seed", required=True, help="Seed dictionary CSV")
parser.add_argument("--out-dir", default="data/processed", help="Output folder")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "warmth_axes.json": {"Sociability", "Morality"},
    "competence_axes.json": {"Ability", "Agency"},
}

nltk.download("wordnet", quiet=True)
if wn.get_version() != "3.0":
    raise ValueError("Seed sense numbers require WordNet 3.0.")

with open(args.axes, encoding="utf-8") as f:
    axes = json.load(f)
with open(args.seed, encoding="utf-8-sig", newline="") as f:
    seed = list(csv.DictReader(f))

for filename, dimensions in CATEGORIES.items():
    directions = {}
    for row in seed:
        if row["Dictionary"] in dimensions and row["PoS"] == "ADJECTIVE":
            term = row["term"].replace(" ", "_")
            syn = wn.synsets(term, pos=wn.ADJ)[int(row["sense"]) - 1]
            direction = row["Dir"].strip().lower()
            if direction not in {"low", "high"}:
                raise ValueError(f"Invalid direction for {term}: {direction}")
            directions.setdefault(syn.name(), set()).add(direction)

    output, skipped = {}, []
    for key, text in axes.items():
        fields = dict(
            line.strip().split(": ", 1)
            for line in text.splitlines() if line.strip()
        )
        a, b = fields["Antonym Pair"].split("|")

        # Both poles must imply the same direction for a (the first synset).
        votes = directions.get(a, set()) | {
            "low" if d == "high" else "high"
            for d in directions.get(b, set())
        }
        if not votes:
            continue
        if len(votes) > 1:
            skipped.append(f"{a}|{b}")
            continue

        low, high = (0, 1) if votes == {"low"} else (1, 0)
        names = (a, b)
        poles = (fields["Negative"], fields["Positive"])  # as ordered in the axes file
        domains = fields["Domain"].split(" - ", 1)
        confidence = fields["Confidence"].split(" - ", 1)

        output[key] = (
            f"Antonym Pair: {names[low]}|{names[high]}\n"
            f"  Negative: {poles[low]}\n"
            f"  Positive: {poles[high]}\n"
            f"  Domain: {domains[low]} - {domains[high]}\n"
            f"  Confidence: {confidence[low]} - {confidence[high]}\n"
        )

    path = out_dir / filename
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{path}: {len(output)} axes; skipped conflicts: {skipped}")