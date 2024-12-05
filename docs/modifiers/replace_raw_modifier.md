# `ReplaceRawModifier`

## Description

This modifier replaces the raw text version on the target layer with a
corrected string and propagates the changes to the tokenized version.

## Required

One or more mapping files. A mapping file must be a csv or tsv file that contains at least the columns `basename` and `par_idx` to uniquely identify and join records from the input data and the mapping file(s) and a column `<raw_label>` (e.g. `norm_corrected`) which contains the corrected raw version. Other columns will be ignored.

```bash
cat mapping.csv
"abschatz_gedichte_1704",2916,"Ich weine / lieben Freunde / um mich und euch / Und um das ganze Land zugleich."
```

## Usage

```bash
python3 src/transnormer_data/cli/modify_dataset.py \
    -m replacerawmodifier \
    --modifier-kwargs "mapping_files=<file-path>+ layer={norm,orig} raw_label=<string>" \
    --data <dir-path-in> \
    -o <dir-path-out> &
```
