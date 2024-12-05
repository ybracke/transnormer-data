# `ReplaceToken1toNModifier`

## Description

Modifier that replaces unigrams with ngrams.

This modifier replaces the occurrences of a unigram type on the tokenized version of the target layer (here "norm_tok" or "orig_tok") with an ngram, propagates the changes to the raw version ("norm" or "orig") and computes a new alignment with the source layer ("orig_tok" or "norm_tok", respectively).

Default target layer is `norm`.

For related uses:

* Replacements on the target layer where a single token is replaced by a unigram can be performed with the faster [`ReplaceToken1to1Modifier`](./replace_token_1to1_modifier.md).
* Replacements that depend on the source layer and/or depend on n-grams can be performed with the [`ReplaceNtoMCrossLayerModifier`](./replace_ntom_cross_layer_modifier.md) (much slower).


## Required


## Usage

```bash
$ python3 src/transnormer_data/cli/modify_dataset.py \
    -m replacetoken1tonmodifier \
    --modifier-kwargs "mapping_files=<file-path>+ layer={orig,norm}" \
    --data <dir-path-in> \
    -o <dir-path-out> &
```
