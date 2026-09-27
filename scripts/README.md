### `build_antonym_axes.py`

Constructs the 1,999 adjective antonym axes from WordNet 3.0 using
NLTK and adds BabelDomains labels where available.

- **Requirements:** Python 3.8+ and NLTK.
- **Input:** `babeldomains_wordnet.txt`, containing BabelDomains
  labels for WordNet 3.0 synsets (Camacho-Collados & Navigli, 2017).
  Available at: http://lcl.uniroma1.it/babeldomains/
- **Outputs:**
  - `antonym_axes.json`: Each entry contains an axis identifier
    formed from two WordNet IDs, the two synsets and their lemmas
    and definitions, and each pole's domain and confidence.
    Missing annotations are recorded as `Unknown`.
- **Ordering:** `Pole_A` is the synset with the smaller WordNet
  offset; `Pole_B` is the other synset. Entries are sorted by
  the numeric offsets of `Pole_A`, then `Pole_B`. This provides
  reproducible ordering for fixed inputs and resource versions.
  Pole labels indicate orientation only, not evaluative valence.

Run:

```bash
mkdir -p data/processed
python scripts/build_antonym_axes.py --babel data/raw/babeldomains_wordnet.txt --out data/processed/antonym_axes.json
```
### `build_scm_axes.py`

Constructs Warmth (Sociability + Morality) and Competence
(Ability + Agency) axes by matching WordNet synsets to the
Nicolas et al. (2021) seed dictionary.

- **Requirements:** Python 3.8+. 
- **Inputs:** `data/processed/antonym_axes.json`, containing the WordNet
  antonym axes, and `data/raw/Seed_Dictionaries.csv`, containing seed
  terms, sense numbers, dimensions, and low/high directions.
  
- **Outputs:**
  - `warmth_axes.json`: Axes associated with Sociability or Morality.
  - `competence_axes.json`: Axes associated with Ability or Agency.
  Each entry preserves the original axis ID, synsets, lemmas,
  definitions, domains, and confidence values.
- **Selection:** An axis is retained when at least one pole
  matches a seed synset in the relevant category. If only one
  pole is labeled, its antonym receives the opposite direction.
  Conflicting assignments are reported and excluded.
- **Ordering:** The negative pole is the low pole; the positive
  pole is the high pole. Synsets and pole metadata are ordered
  from negative to positive. Original axis IDs and entry order
  are preserved.

Run:

```bash
python scripts/build_scm_axes.py --axes data/processed/antonym_axes.json --seed data/raw/Seed_Dictionaries.csv --out-dir data/processed
```
