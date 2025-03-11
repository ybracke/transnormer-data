# transnormer-data

Data preparation for the [Transnormer](https://github.com/ybracke/transnormer) project.

## Description

* Tools for creating datasets in the [target format](#Format)
* Tools for modifying datasets, e.g. changing the normalization or adding additional annotations

This project is for modifying and formatting parallel corpora to prepare them as training or test data with the tool [Transnormer](https://github.com/ybracke/transnormer) for historical spelling normalization.

Training examples for Transnormer must conform to the following JSON template:
```jsonc
{
    "orig" : "Eyn Theylſtueck", // original spelling
    "norm" : "Ein Teilstück"    // normalized spelling
}
```

This project facilitates the transformation of corpus data from a variety of input formats into a JSONL format that contains these two attributes and a few more. The full output format is described in the section [Format](#Format).

Preparing a corpus for training and evaluation can entail more than converting the format to JSONL. For example, if the initial corpus only comes in tokenized form, sentences must be converted to a raw string version (e.g. `["Eyn", "Theylstueck"] -> "Eyn Theylstueck"`).

Moreover, this project makes it possible to apply updates to the data, particularly, to the normalization layer. For example, some normalizations in *Deutsches Textarchiv* (DTA, German Text Archive) contain underscores (e.g. `geht_es`), which should be removed. Another case are systematic errors in the normalizations that we want to correct (e.g. German spelling according to pre-1996 rules, e.g. `Kuß` instead of `Kuss`).

*Transnormer* takes raw strings as input, but often we want to apply automatic replacements on the token-level (e.g. replacing `Kuß` by `Kuss`). Thus both versions - a tokenized and a raw string (untokenized) version - of the examples should be kept. Any changes to either version (raw or tokenized) must be propagated to the other, so that they are always in sync. This functionality is provided by the code in this repo.

## Format

This is the format that each dataset is converted to. Attributes that are commented out are optional.

```jsonc
{
    "basename" : str, // identifier of document
    "par_idx" : str,  // index of sentence within document
                      // the pair (basename, par_idx) uniquely identifies an example
    "date" : int,
    "genre" : str,
    "author" : str,
    "title" : str,
    "orig" : str, // raw string version of original text
    "orig_tok" : List[str], // tokenized version of orig
    "orig_ws" : List[bool], // whitespace information: is there ws in front of token i
    "orig_spans" : List[Tuple[int,int]], // character spans of tokens in orig
    // "orig_lemma" : List[str], // lemmas of orig tokens
    // "orig_pos" : List[str],   // part-of-speech of orig tokens
    "norm" : str,
    "norm_tok" : List[str],
    "norm_ws" : List[bool],
    "norm_spans" : List[Tuple[int,int]],
    // "norm_lemma" : List[str],
    // "norm_pos" : List[str],
    "alignment" : List[Tuple[int,int]], // mapping of tokens in orig_tok to tokens in norm_tok
    // extendable with attributes
    // e.g. additional annotations of tokens like NER, foreign material
    // or annotations for the entire example, e.g. a quality score
}
```

## Example dataset

A published dataset in the specified format can be found on Hugging Face: [dtak-transnormer-full-v1](https://huggingface.co/datasets/ybracke/dtak-transnormer-full-v1).

The full dataset (8.3 GB) can be downloaded with the Hugging Face [`CLI`](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```sh
huggingface-cli download ybracke/dtak-transnormer-full-v1 --repo-type dataset --local-dir .
```

## Docs

Any modifications to the source side of the parallel corpus should be done by the `Maker`. After the dataset was made with the `Maker`, no more edits to source are expected. This way, any annotations that belong to the pre-tokenized inputs are preserved (or have to be handled specifically by the `Maker`).

Modifications to the target side of the corpus can be applied with a given `Modifier` during creation of a dataset and to an existing (previously created) dataset.

The target side's tokenization may change during modifications. At all times, we must be able to create `target_raw` from `target_tok` and vice versa. For this reason, we create a single source of truth - `target_raw` - during dataset creation with the `Maker`.

### Makers

A `Maker` defines how to create a corpus in the [target format](#format) from the input data and metadata files. This involves tokenization or detokenization (depending on the input format), alignmend, span computation, etc.

The metadata and data files input formats must be accounted for by the `Maker`.

Currently, this project supports the following makers:

* `DtaEvalMaker`: for the DTAEvalCorpus (data format: custom XML)
* `DtakMaker`: for the DTAK and DTAE Corpus (data format: ddctabs)

### Modifiers

A `Modifier` defines one or more modifications per record in the data.
A modification can be the update of an existing property (e.g. replace a token in the tokens list with `ReplaceToken1to1Modifier`) and propagate the changes to other prorties (e.g. recompute the token alignment) or the addition of a new property (e.g. adding a language model score with `LMScoreModifier`).

Currently, this project supports the following modifiers:

* [`LanguageDetectionModifier`](docs/modifiers/language_detection_modifier.md)
* [`LanguageToolModifier`](docs/modifiers/language_tool_modifier.md)
* [`LMScoreModifier`](docs/modifiers/lm_score_modifier.md)
* [`ReplaceNtoMCrossLayerModifier`](docs/modifiers/replace_ntom_cross_layer_modifier.md)
* [`ReplaceRawModifier`](docs/modifiers/replace_raw_modifier.md)
* [`ReplaceToken1to1Modifier`](docs/modifiers/replace_token_1to1_modifier.md)
* [`ReplaceToken1toNModifier`](docs/modifiers/replace_token_1ton_modifier.md)

### Script `dataset2lexicon.py`

This script creates a lexicon of ngram alignments between original and normalized ngrams from a dataset that has been processed with a Maker and/or a Modifier (see above). The ngram alignments in the output lexicon are drawn from sentence-level ngram alignments that have already been computed and are stored in the dataset.
Therefore, datasets that are passed to this script must be one or more JSONL files that at least contain the properties `'orig_tok'`, `'norm_tok'` and `'alignment'`.
The output is a lexicon JSONL file that has the following properties for each orig-norm pair:
- `ngram_orig`: ngram in original spelling
- `ngram_norm`: ngram in normalized spelling that is aligned to `ngram_orig`
- `freq`: total frequency of the pair `(ngram_orig, ngram_norm)` in the dataset
- `docs`: document frequency of the pair, i.e. in how many documents does it occur

More on the ngram alignment: The token-level alignments produced with [textalign](https://github.com/ybracke/textalign) are n:m alignments, and aim to be the best alignment between the shortest possible sequence of tokens on the layer `orig_tok` with the shortest possible sequence of tokens on the layer `norm_tok`. Therefore, most of the mappings will be 1:1 alignments, followed by 1:n/n:1 alignments.

```
python3 src/transnormer_data/cli/dataset2lexicon.py --help

usage: dataset2lexicon.py [-h] --data DATA -o OUT [-s NGRAM_SEPARATOR] [-x] [--keep-none-alignments]

Creates a lexicon from a dataset with the following fields: `ngram_orig`, `ngram_norm`, `freq`, `docs`

options:
  -h, --help            show this help message and exit
  --data DATA           Path to the input data file or directory, or a glob path.
  -o OUT, --out OUT     Path to the output file (JSONL).
  -s NGRAM_SEPARATOR, --ngram_separator NGRAM_SEPARATOR
                        Separator symbol(s) for ngrams that consists of multiple tokens
                        (default='▁'(U+2581)).
  -x, --transliterate   Passing this flag will transliterate the 'orig_tok' layer before counting (e.g. this
                        would merge the spellings 'ſchoͤn' and 'schön', since 'ſchoͤn' gets converted to
                        'schön').
  --keep-none-alignments
                        Passing this flag to include None aligments in the lexicon, that is, tokens that
                        have not been aligned with any token on the other layer. Note: The current version
                        of this script groups all None alignments for a sentence together into a pseudo-
                        ngram, even if they are do not all not occur in consecutive order.
```

## Installation

```sh
# Download code
git clone git@github.com:ybracke/transnormer-data.git
cd ./transnormer-data

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install --upgrade pip
pip install .
```

## Usage

Start the virtual environment and run the CLI scripts `make_dataset.py`, `split_dataset.py` or `modify_dataset.py`.

Here are some example calls:

```bash
python3 src/transnormer_data/cli/make_dataset.py \
--maker dtakmaker \
--data dta/ddc_tabs/dtak/corpus-tabs.d/ \
--metadata dta/metadata/jsonl/metadata_dtak.jsonl \
--output-dir dta/jsonl/v01/
```

```bash
python3 src/transnormer_data/cli/split_dataset.py \
--input-dir dta/jsonl/v01/ \
--output-dir-train dta/jsonl/v01/1700-1799/train \
--output-dir-validation dta/jsonl/v01/1700-1799/validation \
--output-dir-test dta/jsonl/v01/1700-1799/test \
--year-start 1700 --year-end 1799
```

```bash
python3 src/transnormer_data/cli/modify_dataset.py \
-m replacetoken1to1modifier \
--modifier-kwargs "mapping_files=replacement-dict-1to1.csv layer=norm" \
--data dta/jsonl/v01 \
-o dta/jsonl/v02
```

```bash
python3 src/transnormer_data/cli/modify_dataset.py \
-m replacetoken1to1modifier \
--modifier-kwargs "mapping_files=replacement-dict-1to1.csv layer=norm" \
--data dta/jsonl/v01 \
--o dta/jsonl/v01/dta_all-in-one.jsonl
--output-single-file
```
