import csv
import pytest
import unittest

import datasets

from transnormer_data.modifier.replace_token_1to1_modifier import (
    ReplaceToken1to1Modifier,
)


class ReplaceToken1to1ModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = ReplaceToken1to1Modifier(mapping_files=[])
        self.mapping_files = ["tests/testdata/type-replacements/old2new.tsv"]

    def tearDown(self) -> None:
        pass

    def test_load_mapping_single_file(self) -> None:
        target_mapping = {
            "daß": "dass",
            "muß": "muss",
            "Schiffahrt": "Schifffahrt",
        }
        mapping_files = ["tests/testdata/type-replacements/old2new.tsv"]
        mapping = self.modifier._load_replacement_mapping(mapping_files)
        assert target_mapping == mapping

    def test_load_mapping_two_files(self) -> None:
        target_mapping = {
            "daß": "dass",
            "muß": "muss",
            "Schiffahrt": "Schifffahrt",
            "sehn": "sehen",
            "glühn": "glühen",
        }
        mapping_files = [
            "tests/testdata/type-replacements/old2new.tsv",
            "tests/testdata/type-replacements/error2correct.tsv",
        ]
        mapping = self.modifier._load_replacement_mapping(mapping_files)
        assert target_mapping == mapping

    def test_load_mapping_broken_file(self) -> None:
        mapping_files = ["tests/testdata/type-replacements/broken.tsv"]
        with pytest.raises(csv.Error):
            _ = self.modifier._load_replacement_mapping(mapping_files)

    def test_map_tokens(self) -> None:
        mapping_files = [
            "tests/testdata/type-replacements/old2new.tsv",
            "tests/testdata/type-replacements/error2correct.tsv",
        ]
        self.modifier.type_mapping = self.modifier._load_replacement_mapping(
            mapping_files
        )

        input = [
            "daß",
            "es",
            "heute",
            "in",
            "der",
            "Schiffahrt",
            "noch",
            "so",
            "sein",
            "muß",
            ",",
            "finde",
            "ich",
            "unerträglich",
            "zu",
            "sehn",
        ]
        target = [
            "dass",
            "es",
            "heute",
            "in",
            "der",
            "Schifffahrt",
            "noch",
            "so",
            "sein",
            "muss",
            ",",
            "finde",
            "ich",
            "unerträglich",
            "zu",
            "sehen",
        ]
        result, any_changes = self.modifier.map_tokens(input)
        assert any_changes is True
        assert result == target

        input = ["hier", "passiert", "nichts"]
        target = ["hier", "passiert", "nichts"]
        result, any_changes = self.modifier.map_tokens(input)
        assert any_changes is False
        assert result == target

    def test_modify_sample(self) -> None:
        mapping_files = [
            "tests/testdata/type-replacements/old2new.tsv",
            "tests/testdata/type-replacements/error2correct.tsv",
        ]
        self.modifier.type_mapping = self.modifier._load_replacement_mapping(
            mapping_files
        )

        input_sample = {
            "norm": "daß es heute in der Schiffahrt.",
            "norm_tok": ["daß", "es", "heute", "in", "der", "Schiffahrt", "."],
            "norm_ws": [False, True, True, True, True, True, False],
            "norm_spans": [
                [0, 3],
                [4, 6],
                [7, 12],
                [13, 15],
                [16, 19],
                [20, 30],
                [30, 31],
            ],
        }
        target = {
            "norm": "dass es heute in der Schifffahrt.",
            "norm_tok": ["dass", "es", "heute", "in", "der", "Schifffahrt", "."],
            "norm_ws": [False, True, True, True, True, True, False],
            "norm_spans": [
                [0, 4],
                [5, 7],
                [8, 13],
                [14, 16],
                [17, 20],
                [21, 32],
                [32, 33],
            ],
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target

    def test_modify_dataset(self) -> None:
        mapping_files = ["tests/testdata/type-replacements/old2new.tsv"]
        data_files = ["tests/testdata/jsonl/dtak/varnhagen_rahel01_1834.jsonl"]
        mapping = self.modifier._load_replacement_mapping(mapping_files)
        dataset = datasets.load_dataset("json", data_files=data_files, split="train")
        self.modifier.type_mapping = mapping
        dataset_mod = self.modifier.modify_dataset(dataset)
        assert (
            dataset[9]["norm"]
            == "Diese Bezeichnung darf indes auch jetzt, da jenem Verlangen nachgegeben wird, im vollen Sinne fortdauern; denn noch immer sind es wesentlich die Freunde, für welche der neue Abdruck Stadt findet, nur daß den im Leben gekannten jetzt auch die nach dem Scheiden erworbenen und künftigen sich anschließen."
        )
        assert (
            dataset_mod[9]["norm"]
            == "Diese Bezeichnung darf indes auch jetzt, da jenem Verlangen nachgegeben wird, im vollen Sinne fortdauern; denn noch immer sind es wesentlich die Freunde, für welche der neue Abdruck Stadt findet, nur dass den im Leben gekannten jetzt auch die nach dem Scheiden erworbenen und künftigen sich anschließen."
        )


class ReplaceToken1to1ModifierTesterOrigLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = ReplaceToken1to1Modifier(layer="orig", mapping_files=[])
        self.mapping_files = ["tests/testdata/type-replacements/old2new.tsv"]

    def tearDown(self) -> None:
        pass

    def test_modify_sample(self) -> None:
        mapping_files = [
            "tests/testdata/type-replacements/old2new.tsv",
            "tests/testdata/type-replacements/error2correct.tsv",
        ]
        self.modifier.type_mapping = self.modifier._load_replacement_mapping(
            mapping_files
        )

        input_sample = {
            "orig": "daß es heute in der Schiffahrt.",
            "orig_tok": ["daß", "es", "heute", "in", "der", "Schiffahrt", "."],
            "orig_ws": [False, True, True, True, True, True, False],
            "orig_spans": [
                [0, 3],
                [4, 6],
                [7, 12],
                [13, 15],
                [16, 19],
                [20, 30],
                [30, 31],
            ],
        }
        target = {
            "orig": "dass es heute in der Schifffahrt.",
            "orig_tok": ["dass", "es", "heute", "in", "der", "Schifffahrt", "."],
            "orig_ws": [False, True, True, True, True, True, False],
            "orig_spans": [
                [0, 4],
                [5, 7],
                [8, 13],
                [14, 16],
                [17, 20],
                [21, 32],
                [32, 33],
            ],
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target
