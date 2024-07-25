# transnormer-data

Data preparation for the [transnormer](https://github.com/ybracke/transnormer) project. 

## Short description

* Tools for creating datasets in the [target format](#Format)
* Tools for modifying datasets, e.g. changing the normalization or adding additional annotations  
* [Planned] Tool for checking if files adhere to the [target format](#Format)

## Background

This repository provides code for formatting, modifying and storing corpus data that is used by the [transnormer](https://github.com/ybracke/transnormer) project, which aims to train a transformer model for historical spelling normalization.

The purpose of this repository is to facilitate and standardize the data preparation across different datasets that are used for training and evaluating the transnormer.

Training examples for transnormer must have a simple JSON structure: 
```jsonc
{
    "orig" : "Eyn Theylſtueck", // original spelling
    "norm" : "Ein Teilstück"    // normalized spelling
}
```
This repository supports to transform corpus data from a variety of input formats into a JSONL format that contains these two attributes. The JSONL format that is created by the code in this repository (see [below](#Format)) is more detailed than the minimal version shown above, for reasons outlined in the following. 

Preparing the data for training and evaluation typically entails more than converting the format to JSONL. For example, the corpus data usually comes in tokenized form and must be converted to a raw string version (e.g. `["Eyn", "Theylstueck"] -> "Eyn Theylstueck"`). 

Moreover, there can be the need for improvements to the data. For example, normalizations in Deutsches Textarchiv (DTA) may contain underscores (e.g. `geht_es`), which should be removed. There may also be systematic errors in the normalizations that we want to correct (e.g. German spelling according to pre-1996 rules, e.g. `daß` instead of `dass`). This repository provides code for doing automatic modifications in the corpus data to improve its quality.

As seen above, the transnormer takes raw strings as input, but automatic replacements often operate on tokens/types (e.g. replacing `daß` by `dass`). That is why it makes sense to keep both, a tokenized and a raw string (untokenized) version of the examples. Any changes to either version (raw or tokenized) must be propagated to the other, so that they are always in sync. This functionality is provided by the code in this repo.

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

## Usage [Planned]

Write a `Maker` or `Modifier` object based on `BaseDatasetMaker` (**TODO**) and/or  `BaseDatasetModifier`.

Run the CLI scripts `make_dataset.py` or `modify_dataset.py` by using the respective `Maker` or `Modifier` object as a plugin (**TODO**).

```bash
make_dataset.py --plugin X --input-dir Y --output-dir Z 

modify_dataset.py --plugin X --input-dir Y -output-dir Z
```


## API documentation [Planned]

See the [docs/](docs)folder (**TODO**)


## Modifiers

### LanguageDetectionModifier


Required: 

1. Install protobuf compiler for pycld3
`sudo apt install -y protobuf-compiler`

2. Download model for fastText
```bash
mkdir resources
wget -nc -q "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -O "resources/lid.176.ftz"
```


### LMScoreModifier

Required:

```
conda install -y pip
conda create -y --name <environment-name> python=3.10 pip
conda activate <environment-name>

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```


Example call:

```
conda activate <environment-name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_VISIBLE_DEVICES=1
python3 src/transnormer_data/cli/modify_dataset.py \
    -m lmscoremodifier \
    --modifier-kwargs "model=dbmdz/german-gpt2 layer=norm" \
    --data /home/bracke/data/dta/jsonl/v07/arends_exter_1708.jsonl \
    -o /home/bracke/data/dta/jsonl/v08/ &
```