"""Build warmth and competence axes from the Nicolas et al. (2021) seed dictionary.

An axis from antonym_axes.json is kept if at least one pole (synset) is in the seed
dictionary for the category; the other pole gets the opposite direction.
Axes whose poles get conflicting directions are skipped.
Negative = low pole, Positive = high pole. Entries keep the order of antonym_axes.json.
"""
import csv, json
import nltk
from nltk.corpus import wordnet as wn

AXES, SEED = "antonym_axes.json", "Seed_Dictionaries.csv"
CATEGORIES = {"warmth_axes.json": {"Sociability", "Morality"},
              "competence_axes.json": {"Ability", "Agency"}}

nltk.download("wordnet", quiet=True)
assert wn.get_version() == "3.0", "sense numbers need WordNet 3.0"
axes = json.load(open(AXES, encoding="utf-8"))
seed = list(csv.DictReader(open(SEED, encoding="utf-8-sig")))

for out_file, dims in CATEGORIES.items():
    # synset name -> set of directions ("high"/"low") in this category
    dirs = {}
    for r in seed:
        if r["Dictionary"] in dims and r["PoS"] == "ADJECTIVE":
            syn = wn.synsets(r["term"].replace(" ", "_"), pos=wn.ADJ)[int(r["sense"]) - 1]
            dirs.setdefault(syn.name(), set()).add(r["Dir"])

    out, skipped = {}, []
    for key, text in axes.items():
        pair, pole_a, pole_b, domain, conf = [l.split(": ", 1)[1] for l in text.splitlines()]
        a, b = pair.split("|")
        # direction of A: its own label, or the opposite of B's label
        votes = set(dirs.get(a, set())) | {"low" if d == "high" else "high" for d in dirs.get(b, set())}
        if not votes:
            continue                                   # axis not in this category
        if len(votes) > 1:
            skipped.append(pair); continue             # conflicting labels
        d = domain.split(" - "); c = conf.split(" - ")
        lo, hi = (0, 1) if votes == {"low"} else (1, 0)  # index of low / high pole
        names, poles = (a, b), (pole_a, pole_b)
        out[key] = (f"Antonym Pair: {names[lo]}|{names[hi]}\n"
                    f"  Negative: {poles[lo]}\n"
                    f"  Positive: {poles[hi]}\n"
                    f"  Domain: {d[lo]} - {d[hi]}\n"
                    f"  Confidence: {c[lo]} - {c[hi]}\n")

    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{out_file}: {len(out)} axes, skipped (conflict): {skipped}")
