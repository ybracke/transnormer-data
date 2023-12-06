import os
from typing import List
import unittest

import datasets

from transnormer_data.maker.dtak_maker import DtakMaker

TMPDIR = "tests/testdata/tmp"

class DtaEvalMakerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir_data = "tests/testdata/dtak/ddctabs/"
        self.input_dir_meta = "tests/testdata/metadata/metadata_dtak.jsonl"
        self.output_dir = TMPDIR
        self.maker_configs = {}
        self.target_properties = {
            "basename": str,
            "par_idx": str,
            # "date": int,
            # "genre": str,
            # "author": str,
            # "title": str,
            # "orig": str,
            "orig_tok": List[str],
            "orig_xlit" : List[str], 
            "orig_lemma" : List[str], 
            "orig_pos" : List[str], 
            "orig_ws": List[bool],
            # "orig_spans": List[List[int]],
            # "norm": str,
            "norm_tok": List[str],
            "norm_ws": List[bool],
            # "norm_spans": List[List[int]],
            # "alignment": List[List[int]],
        }

        self.maker = DtakMaker(
            self.input_dir_data, self.input_dir_meta, self.output_dir
        )
        # self.dataset: datasets.Dataset = self.maker.make(save=True)

    def tearDown(self) -> None:
        # Remove files that were created during training
        for root, dirs, files in os.walk(TMPDIR, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            else:
                os.rmdir(root)

    def test_load_data(self) -> None:
        dataset = self.maker._load_data()
        assert dataset.features.keys() == self.target_properties.keys()

    @unittest.skip 
    def test_file_creation(self) -> None:
        """Test whether correctly named files have been created by the Maker"""
        path_01 = f"{TMPDIR}/varnhagen_rahel01_1834.jsonl"
        path_02 = f"{TMPDIR}/weigel_moralweissheit_1674.jsonl"
        assert os.path.isfile(path_01)
        assert os.path.isfile(path_02)
    
    @unittest.skip 
    def test_data_structure(self) -> None:
        """Test whether the dataset's features are identical to the target features"""
        assert self.dataset.features.keys() == self.target_properties.keys()

    @unittest.skip
    def test_shape(self) -> None:
        """Test whether dataset's shape is correct"""
        # assert self.dataset.shape == (28, len(self.target_properties))
        pass

    @unittest.skip
    def test_alignments(self) -> None:
        """Check whether the correct alignments were computed"""
        pass
