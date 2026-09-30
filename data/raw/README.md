# Raw data

## BabelDomains

`babeldomains_wordnet.txt` contains the WordNet domain annotations
used by `scripts/build_antonym_axes.py`.

**Download:** Extract the file from the
[BabelDomains package](http://lcl.uniroma1.it/babeldomains/)
and place it in this folder.

**Source:** José Camacho-Collados and Roberto Navigli (2017).
[BabelDomains: Large-Scale Domain Labeling of Lexical Resources](http://lcl.uniroma1.it/babeldomains/EACL2017_BabelDomains.pdf).
EACL, Short Papers.

## Stereotype seed dictionaries

`Seed_Dictionaries.csv` contains seed terms, parts of speech,
WordNet sense numbers, and low/high directions for seven stereotype dimensions. We use Sociability and Morality for Warmth,
and Ability and Agency for Competence.

**Download:** Download the seed dictionary CSV from
[OSF](https://osf.io/yx45f/overview) and place it in this folder.

**Source:** Gandalf Nicolas, Xuechunzi Bai, and Susan T. Fiske (2021).
[Comprehensive stereotype content dictionaries using a semi-automated method](https://doi.org/10.1002/ejsp.2724).
*European Journal of Social Psychology*.

## Social group terms (WCST)

`WCST-final-dataset.xlsx` contains the 540 social group terms (429 unigrams, 111 multiword)
used for position prediction (RQ1) in `scripts/predict_positions.py`. Sheet `Dev` (270 terms)
and sheet `Test` (270 terms) are stratified by the four Warmth x Competence sign quadrants.
Columns: `term`, `W sign (+1/-1)` (Warmth), `C sign (+1/-1)` (Competence).

We built it from the [Words of Warmth lexicon](https://saifmohammad.com/WebPages/warmth.html):
social group mentions were identified with `claude-fable-5-1` plus manual review, slurs and
multi-sense terms were removed, and the human Warmth and Competence scores were binarized
by their sign.

**Source:** Saif M. Mohammad (2025).
[Words of Warmth: Trust and Sociability Norms for over 26k English Words](https://aclanthology.org/2025.acl-long.922/).
ACL (Volume 1: Long Papers), pages 18830-18850.
