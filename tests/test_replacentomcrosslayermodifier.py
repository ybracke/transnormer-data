import csv
import pytest
import unittest

from typing import Dict, List, Set, Tuple

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

    def test_get_index_map(self) -> None:
        search_tuples = [(0,)]
        alignment = list(zip([0, 0, 1, 2], [0, 1, 2, 3]))
        correct_mapping = {(0,): (0, 1)}
        actual_mapping = self.modifier._get_index_map(search_tuples, alignment)
        assert actual_mapping == correct_mapping

    def test_find_ngram_indices(self) -> None:
        # With ngram lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams: Set[Tuple[str, ...]] = {
            ("daß", "es"),
            ("Schiffahrt", "."),
            ("nicht", "drin"),
        }
        correct_mapping: Dict[Tuple[str, ...], List[Tuple[int, ...]]] = {
            ("daß", "es"): [(0, 1)],
            ("Schiffahrt", "."): [(5, 6)],
        }
        actual_mapping = self.modifier._find_ngram_indices(
            ngrams, sent_tok, ngram_lengths=[2]
        )
        assert actual_mapping == correct_mapping

        # Without providing ngram lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams = {("daß", "es"), ("Schiffahrt", "."), ("nicht", "drin")}
        correct_mapping = {("daß", "es"): [(0, 1)], ("Schiffahrt", "."): [(5, 6)]}
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == correct_mapping

        # Ngrams of different lengths
        sent_tok = ["daß", "es", "heute", "in", "der", "Schiffahrt", "."]
        ngrams = {
            ("daß", "es"),
            ("es", "heute"),
            ("der", "Schiffahrt", "."),
            ("nicht", "drin"),
        }
        correct_mapping = {
            ("daß", "es"): [(0, 1)],
            ("es", "heute"): [(1, 2)],
            ("der", "Schiffahrt", "."): [(4, 5, 6)],
        }
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == correct_mapping

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
        correct_mapping = {
            ("ich", "ich"): [(0, 1), (1, 2)],
            ("ich", "ich", "ich"): [(0, 1, 2)],
            ("immer", "wieder"): [(5, 6), (7, 8)],
        }
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == correct_mapping

        # Unigram
        sent_tok = [
            "wieviel",
            "ist",
            "es",
        ]
        ngrams = {("wieviel",)}
        alignment = [[0, 0], [0, 1], [1, 2], [2, 3]]
        correct_mapping = {("wieviel",): [(0,)]}
        actual_mapping = self.modifier._find_ngram_indices(ngrams, sent_tok)
        assert actual_mapping == correct_mapping

    def test_get_idx2ngram_trg_simple(self) -> None:
        ngrams2indices_src = {("mußt", "’"): [(0, 1), (3, 4)], ("a",): [(2,)]}
        replacement_lex = {("mußt", "’"): ("musste",)}
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3]]
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {
            (0, 1): ("musste",),
            (3,): ("musste",),
        }
        assert actual_target_ngrams2indices == correct_target_ngrams2indices

    def test_get_idx2ngram_trg_empty_sent(self) -> None:
        ngrams2indices_src = {("mußt", "’"): [(0, 1), (3, 4)], ("a",): [(2,)]}
        replacement_lex = {("mußt", "’"): ("musste",)}
        alignment = []  # type: ignore
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {}  # type: ignore
        assert actual_target_ngrams2indices == correct_target_ngrams2indices

    def test_get_idx2ngram_trg_overlap(self) -> None:
        ngrams2indices_src = {
            ("mußt", "’"): [(0, 1), (3, 4)],
            ("’", "n"): [(1, 2)],
        }
        replacement_lex = {
            ("mußt", "’"): ("musste",),
            ("’", "n"): ("ein",),
            ("wieviel",): ("wie", "viel"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5]]
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {
            (0, 1): ("musste",),
            (1, 2): ("ein",),
            (3,): ("musste",),
        }
        assert actual_target_ngrams2indices == correct_target_ngrams2indices

    def test_get_idx2ngram_trg_1ton(self) -> None:
        ngrams2indices_src = {
            ("wieviel",): [(5,)],
        }

        replacement_lex = {
            ("wieviel",): ("wie", "viel"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5], [6, 6]]
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {
            (4, 5): ("wie", "viel"),
        }
        assert actual_target_ngrams2indices == correct_target_ngrams2indices

    def test_get_idx2ngram_trg_1ton_start_end(self) -> None:
        ngrams2indices_src = {
            ("wieviel",): [(0,), (5,)],
        }

        replacement_lex = {
            ("wieviel",): ("wie", "viel"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5]]
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {
            (0,): ("wie", "viel"),
            (4, 5): ("wie", "viel"),
        }
        assert actual_target_ngrams2indices == correct_target_ngrams2indices

    def test_get_idx2ngram_trg_nto1_start_end(self) -> None:
        ngrams2indices_src = {
            ("unter", "halb"): [(0, 1), (4, 5)],
        }

        replacement_lex = {
            ("unter", "halb"): ("unterhalb"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5]]
        actual_target_ngrams2indices = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_ngrams2indices = {
            (0, 1): ("unterhalb"),
            (3, 4, 5): ("unterhalb"),
        }
        assert actual_target_ngrams2indices == correct_target_ngrams2indices