import datasets
import json
import pandas as pd

from datasets import disable_caching
disable_caching()

files = [
    # "/home/bracke/Schreibtisch/schnitzler-25-1.jsonl",
    # "/home/bracke/Schreibtisch/schnitzler-25-2.jsonl",
    "/home/bracke/Schreibtisch/schnitzler-50-1.jsonl",
]
dataset = datasets.load_dataset("json", data_files=files, split="train")

def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return data


data = read_jsonl(files[0])

prev_row = data[0]
max_len_row = -1
for row in data[1:]:
    max_len_row = len(row["orig_tok"]) if len(row) > max_len_row else max_len_row
    assert prev_row.keys() == row.keys()

print(max_len_row)

# dataframe = pd.read_json(files[0], lines=True)

# dataset = datasets.Dataset.from_pandas(dataframe)
# print(dataset)