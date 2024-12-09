# `LMScoreModifier`

## Description

Modifier that adds a language model (LM) probability score to each record.

LM must be a huggingface model, default is "dbmdz/german-gpt2".
By default, LM scores are computed for the layer "norm".

A new property is created for the score with the language model's name, e.g.

```json
{
    "dbmdz/german-gpt2" : 5.6789
}
```


## Required


```
conda install -y pip
conda create -y --name <environment-name> python=3.10 pip
conda activate <environment-name>

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```



## Usage

```bash
conda activate <environment-name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 src/transnormer_data/cli/modify_dataset.py \
    -m lmscoremodifier \
    --modifier-kwargs "model=<hf-model-name> layer={orig,norm}" \
    --data <dir-path-in> \
    -o <dir-path-out> &
```
