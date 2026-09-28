"""Build the probing dataset: labeled template sentences for both poles of every semantic axis.

Usage:
  python 1-data.py --axes wordnet_semantic_axes.json --template-type listing --n 30

Each axis gets its own random generator (seed + axis key), so an axis always gets the
same sentences, regardless of which file it is in or in which order it is processed.
Output: probing_<template-type>_n<n>.json
"""
import argparse
import json
import random

LISTING_TEMPLATES = [
    "List adjectives similar to {ADJ}, in the sense of {DEF}.",
    "Name synonyms of {ADJ}, meaning {DEF}.",
    "Provide words similar to {ADJ} i.e., {DEF}.",
    "Name adjectives that share the meaning of {ADJ}, in the sense of {DEF}.",
    "List adjectives that capture the condition of being {ADJ}, meaning {DEF}.",
    "Provide similar adjectives to {ADJ} i.e., {DEF}.",
    "Name adjectives that reflect the state of being {ADJ}, in the sense of {DEF}.",
    "List closely related adjectives to {ADJ}, meaning {DEF}.",
    "Provide similar words to {ADJ}, in the sense of {DEF}.",
    "Name adjectives that describe manifestations of {ADJ}, meaning {DEF}.",
    "List {ADJ} characteristics i.e., {DEF}.",
    "Provide three synonyms of the adjective {ADJ}, in the sense of {DEF}.",
    "Name adjectives that describe how {ADJ} presents itself, meaning {DEF}.",
    "List adjectives for situations that would be considered {ADJ} i.e., {DEF}.",
    "Name behaviors or traits that exemplify being {ADJ}, in the sense of {DEF}.",
    "Provide adjectives one would use to recognize something as {ADJ}, meaning {DEF}.",
    "List the typical features of something that is {ADJ} i.e., {DEF}.",
    "Name adjectives that reflect {ADJ}, in the sense of {DEF}.",
    "List adjectives that reflect observable qualities of {ADJ}, meaning {DEF}.",
    "Provide adjectives that make something feel distinctly {ADJ} i.e., {DEF}.",
    "Name defining properties commonly linked to {ADJ}, in the sense of {DEF}.",
    "List adjectives that describe how something earns the description {ADJ}, meaning {DEF}.",
    "Provide the qualities implied when something is said to be {ADJ} i.e., ({DEF}).",
    "Name adjectives that convey the nature of {ADJ}, in the sense of {DEF}.",
    "List the shared traits among instances described as {ADJ}, meaning {DEF}.",
    "Provide the defining signals that indicate {ADJ} i.e., {DEF}.",
    "Name the qualities that justify calling something {ADJ}, in the sense of {DEF}.",
    "List the qualities that make an example clearly {ADJ}, meaning {DEF}.",
    "Provide adjectives that create the general impression of {ADJ} i.e., {DEF}.",
    "Name adjectives that describe the defining aspects of {ADJ}, in the sense of {DEF}.",
]

SIMPLE_TEMPLATES = [
    "{ADJ} is: {DEF}.",
    "{ADJ} means: {DEF}.",
    "Definition of {ADJ}: {DEF}.",
    "{ADJ}: {DEF}.",
    "The word {ADJ} means {DEF}.",
    "The adjective {ADJ} is defined as: {DEF}.",
    "{ADJ} refers to: {DEF}.",
    "Meaning of {ADJ}: {DEF}.",
    "{ADJ} — {DEF}.",
    "The term {ADJ} describes: {DEF}.",
    "{ADJ}, i.e., {DEF}.",
    "{ADJ} (meaning {DEF}).",
    "Something {ADJ} is {DEF}.",
    "To be {ADJ} is to be {DEF}.",
    "{ADJ}: characterized by {DEF}.",
    "A person described as {ADJ} is {DEF}.",
    "{ADJ} can be defined as {DEF}.",
    "The concept of {ADJ}: {DEF}.",
    "{ADJ} in one word: {DEF}.",
    "Simply put, {ADJ} means {DEF}.",
    "{ADJ} = {DEF}.",
    "What does {ADJ} mean? {DEF}.",
    "{ADJ}, which means {DEF}.",
    "The meaning of {ADJ} is {DEF}.",
    "If someone is {ADJ}, they are {DEF}.",
    "{ADJ}: essentially {DEF}.",
    "In short, {ADJ} means {DEF}.",
    "{ADJ} — in other words, {DEF}.",
    "To call something {ADJ} is to say it is {DEF}.",
    "{ADJ}, that is, {DEF}.",
]

TEMPLATES = {"listing": LISTING_TEMPLATES, "simple": SIMPLE_TEMPLATES}


def parse_pole(line, label):
    """'  Negative: large, big → definition'  ->  (['large', 'big'], 'definition')"""
    name, rest = line.strip().split(": ", 1)
    assert name == label, f"expected a {label} line, got: {line!r}"
    words, definition = rest.split(" → ", 1)
    return [w.strip().replace("_", " ") for w in words.split(",")], definition.strip()


def sample_sentences(rng, lemmas, definition, templates, n):
    """One sentence per (adjective, template), then sample n of them."""
    sentences = [t.format(ADJ=adj, DEF=definition) for adj in lemmas for t in templates]
    if len(sentences) >= n:
        return rng.sample(sentences, n)
    return sentences + rng.choices(sentences, k=n - len(sentences))


def main():
    ap = argparse.ArgumentParser(description="Build the probing dataset.")
    ap.add_argument("--axes", required=True, help="semantic axes JSON (Negative/Positive lines)")
    ap.add_argument("--template-type", choices=TEMPLATES, default="listing")
    ap.add_argument("--n", type=int, default=30, help="sentences per pole")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: probing_<template-type>_n<n>.json")
    args = ap.parse_args()

    with open(args.axes, encoding="utf-8") as f:
        axes = json.load(f)

    templates = TEMPLATES[args.template_type]
    data = {}
    for key, text in axes.items():
        lines = text.splitlines()
        neg_syn, pos_syn = lines[0].split(": ", 1)[1].split("|")
        neg_lemmas, neg_def = parse_pole(lines[1], "Negative")
        pos_lemmas, pos_def = parse_pole(lines[2], "Positive")

        rng = random.Random(f"{args.seed}:{key}")   # own generator per axis
        examples = (
            [[s, 1] for s in sample_sentences(rng, pos_lemmas, pos_def, templates, args.n)]
            + [[s, -1] for s in sample_sentences(rng, neg_lemmas, neg_def, templates, args.n)]
        )
        rng.shuffle(examples)
        data[key] = {"negative": neg_syn, "positive": pos_syn, "examples": examples}

    out = args.out or f"probing_{args.template_type}_n{args.n}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"template_type": args.template_type, "n": args.n, "seed": args.seed,
                   "axes": data}, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(data)} axes x {2 * args.n} sentences to {out}")


if __name__ == "__main__":
    main()
