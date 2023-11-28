import json
import os
import sys
from typing import List
import unittest

import datasets

from transnormer_data.maker.dta_eval_maker import DtaEvalMaker


class DtaEvalMakerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir_data = "tests/testdata/dtaeval"
        self.input_dir_meta = "tests/testdata/metadata/metadata_dtak.jsonl"
        self.output_dir = "tests/testdata/out/dtaeval/jsonl"
        self.maker_configs = {}
        self.target_properties = {
            "basename": str,
            "par_idx": str,
            "date": int,
            "genre": str,
            "author": str,
            "title": str,
            "is_bad": bool,
            "orig": str,
            "orig_tok": List[str],
            # "orig_class" : List[str], 
            "orig_ws": List[bool],
            "orig_spans": List[List[int]],
            "norm": str,
            "norm_tok": List[str],
            "norm_ws": List[bool],
            "norm_spans": List[List[int]],
            "alignment": List[List[int]],
        }

        self.maker = DtaEvalMaker(
            self.input_dir_data, self.input_dir_meta, self.output_dir
        )
        self.dataset: datasets.Dataset = self.maker.make(save=True)

    def tearDown(self) -> None:
        # Remove files that were created during training
        for root, dirs, files in os.walk("tests/testdata/out", topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            else:
                os.rmdir(root)

    def test_file_creation(self) -> None:
        """Test whether correctly named files have been created by the Maker"""
        path_01 = "tests/testdata/out/dtaeval/jsonl/brentano_kasperl_1838.jsonl"
        path_02 = "tests/testdata/out/dtaeval/jsonl/fontane_stechlin_1899.jsonl"
        assert os.path.isfile(path_01)
        assert os.path.isfile(path_02)

    def test_data_structure(self) -> None:
        """Test whether the dataset's features are identical to the target features"""
        assert self.dataset.features.keys() == self.target_properties.keys()

    def test_shape(self) -> None:
        """Test whether dataset's shape is correct"""
        assert self.dataset.shape == (28, 16)
