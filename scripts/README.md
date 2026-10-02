### `build_antonym_axes.py`
Builds the 1,999 WordNet semantic axes (WordNet 3.0 via NLTK) with BabelDomains labels.
Poles are ordered by WordNet offset, which is a fixed convention with no meaning.

Input: `data/raw/babeldomains_wordnet.txt` 

Run:
```bash
python scripts/build_antonym_axes.py \
    --babel data/raw/babeldomains_wordnet.txt \
    --out data/processed/antonym_axes.json
```

### `build_scm_axes.py`
Selects the Warmth (Sociability + Morality) and Competence (Ability + Agency) semantic axes
with the stereotype seed dictionary of Nicolas et al. (2021), oriented from low to high pole.

Input: `data/processed/antonym_axes.json` (from `build_antonym_axes.py`) and  `data/raw/Seed_Dictionaries.csv` (stereotype seed dictionary of Nicolas et al., 2021)

Run:
```bash
python scripts/build_scm_axes.py \
    --axes data/processed/antonym_axes.json \
    --seed data/raw/Seed_Dictionaries.csv \
    --out-dir data/processed
```
Output: `warmth_axes.json` (57 axes), `competence_axes.json` (43 axes).

### `build_probing_data.py`
Builds the probing dataset: labeled template sentences for both poles of every semantic axis.
Each axis has its own seed, so it always gets the same sentences.

Input: `data/processed/antonym_axes.json`

Run (example: Listing templates, 30 sentences per pole, as in the paper):
```bash
python scripts/build_probing_data.py \
    --axes data/processed/antonym_axes.json \
    --template-type listing --n 30 \
    --out data/processed/probing_listing_n30.json
```
Options: `--template-type listing|simple`, `--n 15|30` (sentences per pole), `--seed 42`.
Output: `probing_<template>_n<n>.json`

### `build_geometric_axes.py`
Recovers the geometric axis of every semantic axis in every attention head and layer.

Input: `data/processed/probing_<template>_n<n>.json` (from `build_probing_data.py`)

Run (example for the probing dataset above):
```bash
python scripts/build_geometric_axes.py \
    --model NousResearch/Meta-Llama-3-8B-Instruct \
    --data data/processed/probing_listing_n30.json \
    --out-dir checkpoints
```
Options: `--model NousResearch/Meta-Llama-3-8B-Instruct|mistralai/Mistral-7B-Instruct-v0.1|Qwen/Qwen3-8B`.
Output: `<model>_<template>_n<n>_{head,layer}_{mean,last}.npz` for that model and configuration.
`hf_model_utils.py` (model loading and head/layer hooks) must be in the same folder.

### `predict_positions.py`
Position prediction of social groups on the Warmth and Competence semantic axes.


Input: `data/raw/WCST-final-dataset.xlsx` (sheets `Dev` and `Test`),
`data/processed/warmth_axes.json`, `competence_axes.json`, `antonym_axes.json`, and the
geometric axes (downloaded automatically from the Hugging Face dataset and cached, or read
from `--axes-dir`).

Run (final results, as in the paper):
```bash
MODELS="Meta-Llama-3-8B-Instruct Mistral-7B-Instruct-v0.1 Qwen3-8B"
python scripts/predict_positions.py extract --models $MODELS
python scripts/predict_positions.py test --models $MODELS \
    --head listing_n15 mean 128 --layer listing_n15 mean 8
```

### `identify_stereotypical_axes.py`

Tests for each semantic axis to determine whether it is stereotypical.
Input: `data/raw/social_groups.txt`, `data/raw/random_phrases.txt`,
`data/raw/stereotypicality_annotations.csv`, `data/processed/antonym_axes.json`,
geometric axes from the Hugging Face repo (or `--axes-dir`).

Run:

```bash
# 1. activations of social groups and random phrases (once per model)
python scripts/identify_stereotypical_axes.py extract
# 2. 100 stratified axes, compared with human labels
python scripts/identify_stereotypical_axes.py test --axes stratified \
    --head listing_n15 mean 128 --layer listing_n15 mean 8
# 3. all 1,999 axes 
python scripts/identify_stereotypical_axes.py test --axes all --correction bh \
    --head listing_n15 mean 128 --layer listing_n15 mean 8
# optional: power analysis
python scripts/identify_stereotypical_axes.py power --axes stratified
```

