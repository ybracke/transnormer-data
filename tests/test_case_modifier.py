import os
import pytest
import unittest

import datasets

from transnormer_data.modifier.case_modifier import (
    CaseModifier,
)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions"
)
class CaseModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        model = "/home/bracke/models/caser/caser-deu-news-2023-11-08"
        self.modifier = CaseModifier(model_name=model)

    def tearDown(self) -> None:
        pass

    def test_modify_batch(self) -> None:
        data_files = ["tests/testdata/jsonl/dtak/varnhagen_rahel01_1834.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        batch = self.dataset[:1]
        batch_mod = self.modifier.modify_batch(batch)
        print(batch_mod["norm"])
        print(batch_mod["norm_tok"])

    def test_modify_dataset_batchsize_1(self) -> None:
        data_files = ["tests/testdata/jsonl/dtak/varnhagen_rahel01_1834.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        self.dataset = self.dataset.select(range(2))
        dataset_mod = self.modifier.modify_dataset(self.dataset, batch_size=1)
        assert dataset_mod is not None
