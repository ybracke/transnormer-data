# `LanguageDetectionModifier`

## Description

This modifier runs language detection algorithms over the raw version of the source or target layer of the corpus and adds the language labels as additional properties to the dataset.

The default layer that the language detection is applied to is `orig`.

In the output, each sample will have four additional properties, like in this example:

```jsonc
{
    "lang_fastText" : "de",   // str (language label fastText)
    "lang_py3langid" : "nl",  // str (language label py3langid)
    "lang_cld3" : "de",       // str (language label cld3)
    "lang_de" : 0.666         // float (rounded to 3 decimal points)
}
```

## Required

1. Install protobuf compiler for pycld3
`sudo apt install -y protobuf-compiler`

2. Download model for fastText
```bash
mkdir resources
wget -nc -q "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -O "resources/lid.176.ftz"
```

## Usage

```bash
python3 src/transnormer_data/cli/modify_dataset.py \
    -m languagedetectionmodifier \
    --data <dir-path-in> \
    -o <dir-path-out>
```
