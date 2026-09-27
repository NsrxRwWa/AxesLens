# AxesLens

Anonymous research materials accompanying the submission
*AxesLens: Probing Concept Representations in LLMs through Semantic Axes*.

### `build_antonym_axes.py`

Constructs the 1,999 adjective antonym axes from WordNet 3.0 using
NLTK and adds BabelDomains labels where available.

- **Requirements:** Python 3.8+ and NLTK. The script downloads the
  WordNet corpus if it is unavailable.
- **Input:** `babeldomains_wordnet.txt`, containing BabelDomains
  labels for WordNet 3.0 synsets (Camacho-Collados & Navigli, 2017).
  Available at: http://lcl.uniroma1.it/babeldomains/
- **Outputs:**
  - `antonym_axes.json`: Each entry contains an axis identifier
    formed from two WordNet IDs, the two synsets and their lemmas
    and definitions, and each pole's domain and confidence.
    Missing annotations are recorded as `Unknown`.
  - `antonym_axes.json.manifest.json`: Resource versions,
    file checksums, axis counts, and the ordering convention.
- **Ordering:** `Pole_A` is the synset with the smaller WordNet
  offset; `Pole_B` is the other synset. Entries are sorted by
  the numeric offsets of `Pole_A`, then `Pole_B`. This provides
  reproducible ordering for fixed inputs and resource versions.
  Pole labels indicate orientation only, not evaluative valence.

Run:

```bash
python build_antonym_axes.py --babel babeldomains_wordnet.txt --out antonym_axes.json
