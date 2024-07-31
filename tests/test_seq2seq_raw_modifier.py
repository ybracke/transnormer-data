import os
import pytest
import unittest

import datasets

from transnormer_data.modifier.seq2seq_raw_modifier import (
    Seq2SeqRawModifier,
)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions"
)
class Seq2SeqRawModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        model = "ybracke/transnormer-dtaeval-v01"
        tokenizer = "ybracke/transnormer-dtaeval-v01"
        self.modifier = Seq2SeqRawModifier(model, tokenizer)

    def tearDown(self) -> None:
        pass

    @pytest.mark.skip()
    def test_modify_batch(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        assert self.dataset is not None
        batch = self.dataset[:2]
        batch_mod = self.modifier.modify_batch(batch)
        print(batch_mod)

    def test_modify_dataset_batchsize_1(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(self.dataset, save_to=False)
        print(dataset_mod[:2])

    def test_modify_dataset_batchsize_2(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(
            self.dataset, batch_size=2, save_to=False
        )
        print(dataset_mod[:2])

    def test_modify_dataset_batchsize_too_large(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(
            self.dataset, batch_size=4, save_to=False
        )
        print(dataset_mod[:2])
