import json
from typing import Any, Callable, Dict, List
import unittest

import datasets

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.modifier.type_replacement_modifier import TypeReplacementModifier

PATH_DATASET = "tests/testdata/testfile_001.jsonl"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return data


def dummy_modification(text: str) -> str:
    return text.upper()


class IdentityReplacementModifier(BaseDatasetModifier):
    def __init__(self, dataset: datasets.Dataset, modification: Callable) -> None:
        # Replacement dictionary
        # Detokenizer?
        # ...

        # Modification Callable
        self.modification_func = modification
        self.layer = "norm"  # TODO
        self.format = "raw"  # TODO
        # self.modification_func_args = fn_args
        super.__init__()

    def modify_dataset(self) -> None:
        """Apply the specified modification(s) to the entire dataset"""
        self.dataset = self.dataset.map(
            self.modify_sample(self.modification_func), fn_kwargs=args, batched=False
        )


class BasicsTester(unittest.TestCase):
    def setUp(self):
        # self.data = read_jsonl(PATH_DATASET)
        self.dataset = datasets.load_dataset("json", data_files=PATH_DATASET)
        self.modifier = TypeReplacementModifier(self.dataset)

    def test_modification(self):
        dataset = self.modifier.modify_dataset()
        assert dataset["train"][0]["norm"] == "Ein tolles Haus."
        assert dataset["train"][0]["norm_tok"] == ["Ein", "tolles", "Haus", "."]
