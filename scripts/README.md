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
Recovers the geometric axis of every semantic axis in every attention head and layer, for
mean-over-tokens and last-token activations, with the variance ratio of each.
Needs a GPU; saves checkpoints and resumes after interruptions.

Input: `data/processed/probing_<template>_n<n>.json` (from `build_probing_data.py`)

Run (example for the probing dataset above):
```bash
python scripts/build_geometric_axes.py \
    --model NousResearch/Meta-Llama-3-8B-Instruct \
    --data data/processed/probing_listing_n30.json \
    --out-dir checkpoints
```
Options: `--model NousResearch/Meta-Llama-3-8B-Instruct|mistralai/Mistral-7B-Instruct-v0.1`.
Output: `head_mean`, `head_last`, `layer_mean`, `layer_last` (`.npz`) for that model and
configuration. 
`pipeline_utils.py` contains the shared model and tokenizer loaders and must be in
the same folder as `build_geometric_axes.py`.

### `predict_positions.py`
Position prediction of social groups on the Warmth and Competence semantic axes (RQ1).
The sign of a group's projection onto a geometric axis gives the predicted label (+1/-1),
For Warmth and Competence the poles must point
from the low pole to the high pole (e.g. unfriendly to friendly). Therefore, it compares each
axis in `warmth_axes.json` / `competence_axes.json` with `antonym_axes.json` and reverses the
geometric axis (multiplies it by -1) wherever the two orders differ. 

Input: `data/raw/WCST-final-dataset.xlsx` (sheets `Dev` and `Test`) and `data/processed/warmth_axes.json`, `competence_axes.json`, `antonym_axes.json` and the geometric axes, downloaded automatically from the Hugging Face dataset and cached

Run:
```bash
python scripts/predict_positions.py test --head listing_n15 mean 128 --layer listing_n15 mean 4
```
