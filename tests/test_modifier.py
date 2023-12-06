import json
from typing import Any, Callable, Dict, List
import unittest

import datasets

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.modifier.type_replacement_modifier import TypeReplacementModifier

PATH_DATASET = "tests/testdata/testfile_001.jsonl"


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


class ModifierTester(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.load_dataset("json", data_files=PATH_DATASET)
        self.modifier = TypeReplacementModifier(self.dataset)

    @unittest.skip
    def test_modification(self):
        # Remove alignment to show that it is successfully recreated
        self.dataset["train"] = self.dataset["train"].remove_columns("alignment")
        print(self.dataset["train"][0])
        self.dataset = self.modifier.modify_dataset()
        assert self.dataset["train"][0]["norm"] == "Ein tolles Haus."
        assert self.dataset["train"][0]["norm_tok"] == ["Ein", "tolles", "Haus", "."]
        assert self.dataset["train"][0]["alignment"] == [[0, 0], [1, 1], [2, 2], [3, 3]]
