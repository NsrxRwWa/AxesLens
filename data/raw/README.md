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

We built it from the [Words of Warmth lexicon](https://saifmohammad.com/WebPages/warmth.html):
social group mentions were identified with `claude-fable-5-1` plus manual review, slurs and
multi-sense terms were removed, and the human Warmth and Competence scores were binarized
by their sign.

**Source:** Saif M. Mohammad (2025).
[Words of Warmth: Trust and Sociability Norms for over 26k English Words](https://aclanthology.org/2025.acl-long.922/).
ACL (Volume 1: Long Papers), pages 18830-18850.

## Stereotypicality identification 

Used for stereotypicality identification in `scripts/identify_stereotypical_axes.py`.

- `social_groups.txt`: the 50 social groups, one per line.
- `random_phrases.txt`: the 50 frequency-matched random phrases, one per line.
- `stereotypicality_annotations_human.csv`: human annotations of the 100 stratified semantic axes
  (one row per axis, 5 annotators). Each axis was rated for one social group.


