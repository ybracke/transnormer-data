import csv
import pytest
import unittest

from transnormer_data.modifier.replace_token_1ton_modifier import (
    ReplaceToken1toNModifier,
)


class ReplaceToken1toNModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = ReplaceToken1toNModifier(mapping_files=[])
        self.mapping_files = ["tests/testdata/type-replacements/1-to-n.tsv"]

    def tearDown(self) -> None:
        pass

    def test_load_mapping_single_file(self) -> None:
        target_mapping = {
            "allzugroß": ["allzu", "groß"],
            "beisammenzusein": ["beisammen", "zu", "sein"],
            "nächstemal": ["nächste", "Mal"],
        }
        mapping_files = ["tests/testdata/type-replacements/1-to-n.tsv"]
        mapping = self.modifier._load_replacement_mapping(mapping_files)
        assert target_mapping == mapping

    def test_load_mapping_broken_file(self) -> None:
        mapping_files = ["tests/testdata/type-replacements/broken.tsv"]
        with pytest.raises(csv.Error):
            _ = self.modifier._load_replacement_mapping(mapping_files)

    def test_map_tokens(self) -> None:
        mapping_files = ["tests/testdata/type-replacements/1-to-n.tsv"]
        self.modifier.type_mapping = self.modifier._load_replacement_mapping(
            mapping_files
        )

        input_tok = [
            "Das",
            "nächstemal",
            "lohnt",
            "es",
            "sich",
            "allzugroß",
            "beisammenzusein",
            ".",
        ]
        input_ws = [False, True, True, True, True, True, True, False]
        target_tok = [
            "Das",
            "nächste",
            "Mal",
            "lohnt",
            "es",
            "sich",
            "allzu",
            "groß",
            "beisammen",
            "zu",
            "sein",
            ".",
        ]
        target_ws = [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]
        result_tok, result_ws, any_changes = self.modifier.map_tokens(
            input_tok, input_ws
        )
        assert any_changes is True
        assert result_tok == target_tok

        input_tok = ["hier", "passiert", "nichts"]
        input_ws = [False, True, True]
        target_tok = ["hier", "passiert", "nichts"]
        target_ws = [False, True, True]
        result_tok, result_ws, any_changes = self.modifier.map_tokens(
            input_tok, input_ws
        )
        assert any_changes is False
        assert result_tok == target_tok
        assert result_ws == target_ws

    def test_modify_sample(self) -> None:
        mapping_files = ["tests/testdata/type-replacements/1-to-n.tsv"]
        self.modifier.type_mapping = self.modifier._load_replacement_mapping(
            mapping_files
        )

        input_sample = {
            "norm": "Das nächstemal lohnt es sich allzugroß beisammenzusein.",
            "norm_tok": [
                "Das",
                "nächstemal",
                "lohnt",
                "es",
                "sich",
                "allzugroß",
                "beisammenzusein",
                ".",
            ],
            "norm_ws": [False, True, True, True, True, True, True, False],
            "norm_spans": [
                [0, 3],
                [4, 14],
                [15, 20],
                [21, 23],
                [24, 28],
                [29, 39],
                [40, 56],
                [56, 57],
            ],
            "orig_tok": [
                "Das",
                "naechſtemal",
                "lohnt",
                "es",
                "ſich",
                "allzugroſz",
                "beyſammenzuſein",
                ".",
            ],
            "alignment": [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
            ],
        }

        target = {
            "norm": "Das nächste Mal lohnt es sich allzu groß beisammen zu sein.",
            "norm_tok": [
                "Das",
                "nächste",
                "Mal",
                "lohnt",
                "es",
                "sich",
                "allzu",
                "groß",
                "beisammen",
                "zu",
                "sein",
                ".",
            ],
            "norm_ws": [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
            "norm_spans": [
                [0, 3],
                [4, 11],
                [12, 15],
                [16, 21],
                [22, 24],
                [25, 29],
                [30, 35],
                [36, 40],
                [41, 50],
                [51, 53],
                [54, 58],
                [58, 59],
            ],
            "orig_tok": [
                "Das",
                "naechſtemal",
                "lohnt",
                "es",
                "ſich",
                "allzugroſz",
                "beyſammenzuſein",
                ".",
            ],
            "alignment": [
                [0, 0],
                [1, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [5, 7],
                [6, 8],
                [6, 9],
                [6, 10],
                [7, 11],
            ],
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target
