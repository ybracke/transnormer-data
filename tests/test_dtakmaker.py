import os
import unittest
from typing import List

import datasets

from transnormer_data.maker.dtak_maker import DtakMaker

TMPDIR = "tests/testdata/tmp"


class DtakMakerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir_data = "tests/testdata/dtak/ddctabs/"
        self.input_dir_meta = "tests/testdata/metadata/metadata_dtak.jsonl"
        self.output_dir = TMPDIR
        self.target_properties = {
            "basename": str,
            "par_idx": str,
            "date": int,
            "genre": str,
            "author": str,
            "title": str,
            "orig": str,
            "orig_tok": List[str],
            "orig_xlit": List[str],
            "orig_lemma": List[str],
            "orig_pos": List[str],
            "orig_ws": List[bool],
            "orig_spans": List[List[int]],
            "norm": str,
            "norm_tok": List[str],
            "norm_ws": List[bool],
            "norm_spans": List[List[int]],
            "alignment": List[List[int]],
        }

        self.maker = DtakMaker(
            self.input_dir_data,
            self.input_dir_meta,
            self.output_dir,
            merge_into_single_dataset=True,
        )
        self.maker.make()
        self.dataset: datasets.Dataset = self.maker._dataset

    def tearDown(self) -> None:
        # Remove files that were created during training
        for root, dirs, files in os.walk(TMPDIR, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            else:
                os.rmdir(root)

    @unittest.skip(
        "From previous stage. Loading is now implicitly tested by test_verify_properties"
    )
    def test_load_data(self) -> None:
        dataset = self.maker._load_data()
        assert dataset.features.keys() == self.target_properties.keys()

    def test_verify_properties(self) -> None:
        """Test whether the dataset's features are identical to the target features"""
        assert self.dataset.features.keys() == self.target_properties.keys()

    def test_file_creation(self) -> None:
        """Test whether correctly named files have been created by the Maker"""
        path_01 = f"{TMPDIR}/varnhagen_rahel01_1834.jsonl"
        path_02 = f"{TMPDIR}/weigel_moralweissheit_1674.jsonl"
        assert os.path.isfile(path_01)
        assert os.path.isfile(path_02)

    def test_verify_shape(self) -> None:
        """Test whether dataset's shape is correct"""
        assert self.dataset.shape == (101, len(self.target_properties))

    def test_verify_data_integrity(self) -> None:
        for i, example in enumerate(self.dataset):
            # Recreate: tok,ws -> raw
            orig_raw_from_tok = self.maker._modifier._tok2raw(
                example["orig_tok"], example["orig_ws"]
            )
            assert example["orig"] == orig_raw_from_tok
            norm_raw_from_tok = self.maker._modifier._tok2raw(
                example["norm_tok"], example["norm_ws"]
            )
            assert example["norm"] == norm_raw_from_tok

            # Recreate: raw <-> tok,ws
            # orig_tok_from_raw, _ = self.maker._modifier._raw2tok(example["orig"])
            # assert example["orig_tok"] == orig_tok_from_raw # Don't do this (retokenization of orig), because we accept the input tokenization of orif as ground truth. Otherwise our input annotations wouldn't match anymore.
            norm_tok_from_raw, _ = self.maker._modifier._raw2tok(example["norm"])
            assert example["norm_tok"] == norm_tok_from_raw

            # All orig annotations have same length
            tok, xlit, lemma, pos, ws, spans = (
                example["orig_tok"],
                example["orig_xlit"],
                example["orig_lemma"],
                example["orig_pos"],
                example["orig_ws"],
                example["orig_spans"],
            )
            assert (
                len(tok) == len(xlit) == len(lemma) == len(pos) == len(ws) == len(spans)
            )

            # All norm annotations have same length
            tok, ws, spans = (
                example["norm_tok"],
                example["norm_ws"],
                example["norm_spans"],
            )
            assert len(tok) == len(ws) == len(spans)

            # Check-alignments
            src_side, trg_side = list(zip(*example["alignment"]))
            # no None alignments
            assert None not in src_side
            assert None not in trg_side
            assert max(src_side) == len(example["orig_tok"]) - 1
            assert max(trg_side) == len(example["norm_tok"]) - 1

    @unittest.skip("Pseudo-test")
    def test_print_dtak_alignments(self) -> None:
        """Pseudo-test that prints some alignments"""
        for i, example in enumerate(
            self.dataset.filter(
                lambda example: example["basename"].startswith("weigel")
            )
        ):
            for aligned_pair in example["alignment"]:
                print(
                    f"{example['orig_tok'][aligned_pair[0]]} <-> {example['norm_tok'][aligned_pair[1]]}"
                )
            print()
            if i > 20:
                break
