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
        model = "/home/bracke/models/caser/caser-deu-news-2023-11-08"
        tokenizer = "/home/bracke/models/caser/caser-deu-news-2023-11-08"
        self.modifier = Seq2SeqRawModifier(model, tokenizer)

    def tearDown(self) -> None:
        pass

    # @pytest.mark.skip()
    def test_modify_batch(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        assert self.dataset is not None
        batch = self.dataset[:2]
        batch_mod = self.modifier.modify_batch(batch)
        print(batch_mod)

    # @pytest.mark.skip()
    def test_modify_dataset_batchsize_None(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        with pytest.raises(NotImplementedError):
            _ = self.modifier.modify_dataset(self.dataset)

    def test_modify_dataset_batchsize_1(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(self.dataset, batch_size=1)
        assert dataset_mod is not None
        print(dataset_mod[:2])

    # @pytest.mark.skip()
    def test_modify_dataset_batchsize_2(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(self.dataset, batch_size=2)
        assert dataset_mod is not None
        print(dataset_mod[:2])

    # @pytest.mark.skip()
    def test_modify_dataset_batchsize_too_large(self) -> None:
        data_files = ["tests/testdata/seq2seq/testfile_001.jsonl"]
        self.dataset = datasets.load_dataset(
            "json", data_files=data_files, split="train"
        )
        dataset_mod = self.modifier.modify_dataset(self.dataset, batch_size=4)
        assert dataset_mod is not None
        print(dataset_mod[:2])
