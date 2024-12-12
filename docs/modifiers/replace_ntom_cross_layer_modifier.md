# `ReplaceNtoMCrossLayerModifier`

## Description

n-gram to m-gram cross layer replacement modifier.

This modifier replaces token sequences in the tokenized target text (here "norm_tok" or "orig_tok") if this token sequence corresponds to a given sequence on the source layer. The changes are propagated to the raw version ("norm" or "orig"), new alignments are computed, etc.

Example: All occurrences of the n-gram (X,Y) on the "orig" layer will get normalized as (X',Y',Z). That is, we exchange the m-gram on the "norm" layer that corresponds to (X,Y) with (X',Y',Z).

### Required

A tsv or csv file with n:m mappings of source sequences (column 1) to target sequences (column2). The elements of a sequence are separated by the space character.

```bash
$ cat mappings.tsv
Sag ' was	Sag was
irgend ' was	irgendetwas
' mal	mal
Neu-York	New York
Nieder-Jagd	Niederjagd
```

## Usage

```bash
nohup nice python3 src/transnormer_data/cli/modify_dataset.py \
    -m replacentomcrosslayermodifier \
    --modifier-kwargs "mapping_files=<file-path>+ delimiter=<delimiter> source_layer={orig,norm} target_layer={norm,orig} [transliterate_source={true,t,yes,1}]" \
    --data <dir-path-in> \
    -o <dir-path-out> &
```

Note:
* If the delimiter is the TAB character, `delimiter={TAB}` must be passed.
* To transliterate the source tokens before dictionary lookup, transliterate_source must be passed with any of `{true,t,yes,1}`
