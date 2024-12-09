# `ReplaceToken1to1Modifier`

## Description

Modifier that replaces unigrams with unigrams.

This modifier replaces the occurrences of a unigram type with another unigram on the tokenized version of the target layer and propagates the changes to the raw version of the target layer.

Default target layer is `norm`.

Note: This modifier works relatively fast, since it only handles 1:1 replacements on the target layer. For related, more complicated uses:

* Replacements on the target layer where a single token is replaced by an ngram can be performed with the [`ReplaceToken1toNModifier`](./replace_token_1ton_modifier.md) (a bit slower).
* Replacements that depend on the source layer and/or depend on n-grams can be performed with the [`ReplaceNtoMCrossLayerModifier`](./replace_ntom_cross_layer_modifier.md) (much slower).


## Required

A csv replacement file, with 1:1 mappings, so no spaces are allowed in column 1 or column2.

bash
```
$ cat 1-to-1-replacements.csv
gehn,gehen
Aderlaß,Aderlass
Zusammenhäng,Zusammenhang
```

## Usage

```bash
$ python3 src/transnormer_data/cli/modify_dataset.py \
    -m replacetoken1to1modifier \
    --modifier-kwargs "mapping_files=<file-path>+ layer={norm,orig}" \
    --data <dir-path-in> \
    -o <dir-path-out> &
```
