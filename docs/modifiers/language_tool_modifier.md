# `LanguageToolModifier`

## Description

This modifier applies the LanguageTool to the raw version of the target layer
and propagates the changes to the tokenized version.

Target layer is fixed to "norm", source layer is fixed to "orig".

Note:
Depending on the data size and number of rules this modification may take a while. For ~150M tokens in 6.1M sentences and ~1000 rules, the script ran for 60 hours.

## Required

### Rules

Select LanguageTool rules to apply and store them by their [id](https://dev.languagetool.org/development-overview#the-basic-elements-of-a-rule) in a one-line-per-rule text format. A rule file may look like this:

```bash
$ cat rules.txt
ZUVIEL
STATT
GENANT_SPELLING_RULE
```

## Usage


```bash
python3 src/transnormer_data/cli/modify_dataset.py \
    -m languagetoolmodifier \
    --modifier-kwargs "rule_file=<file-path-in>" \
    --data <dir-path-in> \
    -o <dir-path-out>
```
