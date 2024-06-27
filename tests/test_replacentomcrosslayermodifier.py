import csv
import pytest
import unittest

from transnormer_data.modifier.replace_ntom_cross_layer_modifier import (
    ReplaceNtoMCrossLayerModifier,
)


class ReplaceNtoMCrossLayerModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = ReplaceNtoMCrossLayerModifier()
        # self.mapping_files = []  # TODO

    def tearDown(self) -> None:
        pass

    def test_find_ngram_indices(self) -> None:
        # With ngram lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams = {("daß", "es"), ("Schiffahrt", "."), ("nicht", "drin")}
        target_mapping = {("daß", "es"): [(0, 1)], ("Schiffahrt", "."): [(5, 6)]}
        actual_mapping = self.modifier._find_ngram_indices(
            ngrams, sent_tok, ngram_lengths=[2]
        )
        assert actual_mapping == target_mapping

        # Without providing ngram lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams = {("daß", "es"), ("Schiffahrt", "."), ("nicht", "drin")}
        target_mapping = {("daß", "es"): [(0, 1)], ("Schiffahrt", "."): [(5, 6)]}
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == target_mapping

        # Ngrams of different lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams = {
            ("daß", "es"),
            ("es", "heute"),
            ("der", "Schiffahrt", "."),
            ("nicht", "drin"),
        }
        target_mapping = {
            ("daß", "es"): [(0, 1)],
            ("es", "heute"): [(1, 2)],
            ("der", "Schiffahrt", "."): [(4, 5, 6)],
        }
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == target_mapping

        # Ngram occurs multiple times of different lengths
        sent_tok = [
            "ich",
            "ich",
            "ich",
            "heißt",
            "es",
            "immer",
            "wieder",
            "immer",
            "wieder",
        ]
        ngrams = {("ich", "ich"), ("ich", "ich", "ich"), ("immer", "wieder")}
        target_mapping = {
            ("ich", "ich"): [(0, 1), (1, 2)],
            ("ich", "ich", "ich"): [(0, 1, 2)],
            ("immer", "wieder"): [(5, 6), (7, 8)],
        }
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == target_mapping
