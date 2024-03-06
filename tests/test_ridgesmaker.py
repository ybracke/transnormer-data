import os
from typing import List
import unittest

import glob

from lxml import etree
import datasets

from transnormer_data import utils

from transnormer_data.maker.ridges_maker import RidgesMaker

TMPDIR = "tests/testdata/tmp"


class RidgesMakerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.input_path_orig = "/home/bracke/data/RIDGES/Paula/RIDGES_Herbology_Version9.0/*/*.dipl.text.xml"
        self.input_path_norm = "/home/bracke/data/RIDGES/Paula/RIDGES_Herbology_Version9.0/*/*.norm.text.xml"
        # self.input_dir_meta = "tests/testdata/metadata/metadata_dtak.jsonl"
        self.output_dir = TMPDIR
        self.target_properties = {
            "basename": str,
            "par_idx": int,
            # "date": int,
            # "genre": str,
            # "author": str,
            # "title": str,
            # "is_bad": bool,
            "orig": str,
            # "orig_tok": List[str],
            # "orig_class" : List[str],
            # "orig_ws": List[bool],
            # "orig_spans": List[List[int]],
            "norm": str,
            # "norm_tok": List[str],
            # "norm_ws": List[bool],
            # "norm_spans": List[List[int]],
            # "alignment": List[List[int]],
        }

        self.maker = RidgesMaker(
            self.input_path_orig,
            self.input_path_norm,
            # self.input_dir_meta, 
            self.output_dir
        )
        self.dataset: datasets.Dataset = self.maker.make(save=True)

    def tearDown(self) -> None:
        # Remove files that were created during training
        for root, dirs, files in os.walk(TMPDIR, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            else:
                os.rmdir(root)

    def test_file_creation(self) -> None:
        """Test whether correctly named files have been created by the Maker"""
        path_01 = f"{TMPDIR}/AlchymistischePractic_1603_Libavius.jsonl"
        assert os.path.isfile(path_01)

    def test_data_structure(self) -> None:
        """Test whether the dataset's features are identical to the target features"""
        assert self.dataset.features.keys() == self.target_properties.keys()

    def test_shape(self) -> None:
        """Test whether dataset's shape is correct"""
        assert self.dataset.shape == (93, len(self.target_properties))

    def test_alignments(self) -> None:
        """Check whether the correct alignments were computed"""
        pass

    def test_create_example_from_s_with_join(self) -> None:
        """Test example creation from a single sentence"""
        pass
