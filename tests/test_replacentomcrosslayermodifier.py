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
        self.mapping_files = ["tests/testdata/type-replacements/2-to-n.tsv"]

    def tearDown(self) -> None:
        pass

    def test_get_index_map(self) -> None:
        search_tuples = [(0,), (0, 1)]
        alignment = list(zip([0, 0, 1, 2], [0, 1, 2, 3]))
        correct_mapping = {(0,): (0, 1), (0, 1): (0, 1, 2)}
        actual_mapping = self.modifier._get_index_map(search_tuples, alignment)
        assert actual_mapping == correct_mapping

    def test_get_index_map_with_None(self) -> None:
        search_tuples = [(0,), (0, 1)]
        alignment = list(zip([0, 0, 1, 2], [0, None, 1, 2]))
        correct_mapping = {(0,): (0,), (0, 1): (0, 1)}
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
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (0, 1): ("musste",),
            (3,): ("musste",),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_idx2ngram_trg_empty_sent(self) -> None:
        ngrams2indices_src = {("mußt", "’"): [(0, 1), (3, 4)], ("a",): [(2,)]}
        replacement_lex = {("mußt", "’"): ("musste",)}
        alignment = []  # type: ignore
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {}  # type: ignore
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_idx2ngram_trg_with_None(self) -> None:
        ngrams2indices_src = {("mußt", "’"): [(0, 1), (3, 4)], ("a",): [(2,)]}
        replacement_lex = {("mußt", "’"): ("musste",)}
        alignment = [[0, 0], [1, None], [2, 1], [None, 2], [3, 3], [3, 4]]
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (0,): ("musste",),
            (3, 4): ("musste",),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

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
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (0, 1): ("musste",),
            (1, 2): ("ein",),
            (3,): ("musste",),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_idx2ngram_trg_1ton(self) -> None:
        ngrams2indices_src = {
            ("wieviel",): [(5,)],
        }

        replacement_lex = {
            ("wieviel",): ("wie", "viel"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5], [6, 6]]
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (4, 5): ("wie", "viel"),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_idx2ngram_trg_1ton_start_end(self) -> None:
        ngrams2indices_src = {
            ("wieviel",): [(0,), (5,)],
        }

        replacement_lex = {
            ("wieviel",): ("wie", "viel"),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5]]
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (0,): ("wie", "viel"),
            (4, 5): ("wie", "viel"),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_idx2ngram_trg_nto1_start_end(self) -> None:
        ngrams2indices_src = {
            ("unter", "halb"): [(0, 1), (4, 5)],
        }

        replacement_lex = {
            ("unter", "halb"): ("unterhalb",),
        }
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [5, 5]]
        actual_target_idxs2ngram = self.modifier._get_idx2ngram_trg(
            ngrams2indices_src, alignment, replacement_lex
        )

        correct_target_idxs2ngram = {
            (0, 1): ("unterhalb",),
            (3, 4, 5): ("unterhalb",),
        }
        assert actual_target_idxs2ngram == correct_target_idxs2ngram

    def test_get_start2ngram_and_end(self) -> None:
        idx2ngram = {
            (0, 1): ("unterhalb",),
            (1, 2): ("musst",),
            (3, 4, 5): ("unterhalb",),
        }
        correct_start2ngram_and_end = {
            0: (("unterhalb",), 1),
            1: (("musst",), 2),
            3: (("unterhalb",), 5),
        }

        actual_start2ngram_and_end = self.modifier._get_start2ngram_and_end(
            idx2ngram, remove_overlap=False
        )
        assert actual_start2ngram_and_end == correct_start2ngram_and_end

        correct_start2ngram_and_end = {
            0: (("unterhalb",), 1),
            3: (("unterhalb",), 5),
        }

        actual_start2ngram_and_end = self.modifier._get_start2ngram_and_end(
            idx2ngram, remove_overlap=True
        )
        assert actual_start2ngram_and_end == correct_start2ngram_and_end

    def test_update_target_tok(self) -> None:
        idx2ngram = {
            (0, 1): ("musst",),
            (1, 2): ("ein",),
            (4, 5): ("einkaufen",),
            (6,): ("mal", "wieder"),
        }
        target_tok_in = ["musst", "’", "n", "Eis", "ein", "kaufen", "malwieder"]
        correct_res = ["musst", "n", "Eis", "einkaufen", "mal", "wieder"]
        actual_res = self.modifier._update_target_tok(target_tok_in, idx2ngram)
        assert correct_res == actual_res

    #  test is disabled because idx2ngram cannot contain None currently
    @pytest.mark.skip
    def test_update_target_tok_with_None(self) -> None:
        idx2ngram = {
            (0, None): ("musst",),
            (1, 2): ("ein",),
            (4, 5): ("einkaufen",),
            (6,): ("mal", "wieder"),
        }
        target_tok_in = ["musst", "’", "n", "Eis", "ein", "kaufen", "malwieder"]
        correct_res = ["musst", "ein", "Eis", "einkaufen", "mal", "wieder"]
        actual_res = self.modifier._update_target_tok(target_tok_in, idx2ngram)
        assert correct_res == actual_res

        idx2ngram = {
            (None, 1): ("FOO",),
            (1, 2): ("ein",),
            (4, 5): ("einkaufen",),
            (6,): ("mal", "wieder"),
        }
        target_tok_in = ["musst", "’", "n", "Eis", "ein", "kaufen", "malwieder"]
        correct_res = ["musst", "FOO", "n", "Eis", "einkaufen", "mal", "wieder"]
        actual_res = self.modifier._update_target_tok(target_tok_in, idx2ngram)
        assert correct_res == actual_res

    def test_load_n2m_replacement_mapping(self) -> None:
        files = ["tests/testdata/type-replacements/2-to-n.tsv"]
        correct_res = {
            ("All", "'"): ("Alle",),
            ("Allein", "'"): ("Alleine",),
            ("Ameiſ", "’"): ("Ameise",),
            ("Andres", "'"): ("Andres", "'"),
        }
        actual_res = self.modifier._load_n2m_replacement_mapping(files, delimiters="\t")
        assert correct_res == actual_res

    def test_map_tokens_cross_layer(self) -> None:
        self.modifier.type_mapping = self.modifier._load_n2m_replacement_mapping(
            self.mapping_files, delimiters="\t"
        )
        tokens_src = ["All", "'", "Andres", "'", "Allein", "'"]
        tokens_trg = ["Alle", "'", "Andres", "'", "Allein", "'"]
        alignment = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        actual_res = self.modifier.map_tokens_cross_layer(
            tokens_src, tokens_trg, alignment
        )
        correct_res = (["Alle", "Andres", "'", "Alleine"], True)
        assert correct_res == actual_res

    def test_modify_sample(self) -> None:
        self.modifier.type_mapping = self.modifier._load_n2m_replacement_mapping(
            self.mapping_files, delimiters="\t"
        )
        input_sample = {
            "norm": "Alle' Andres' Allein'",
            "norm_tok": ["Alle", "'", "Andres", "'", "Allein", "'"],
            "norm_ws": [False, False, True, False, True, False],
            "norm_spans": [
                [0, 4],
                [4, 5],
                [6, 12],
                [12, 13],
                [14, 20],
                [20, 21],
            ],
            "orig_tok": ["All", "'", "Andres", "'", "Allein", "'"],
            "alignment": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        }
        correct_res = {
            "norm": "Alle Andres' Alleine",
            "norm_tok": ["Alle", "Andres", "'", "Alleine"],
            "norm_ws": [False, True, False, True],
            "norm_spans": [
                [0, 4],
                [5, 11],
                [11, 12],
                [13, 20],
            ],
            "orig_tok": ["All", "'", "Andres", "'", "Allein", "'"],
            "alignment": [[0, 0], [1, None], [2, 1], [3, 2], [4, 3], [5, None]],
        }
        actual_res = self.modifier.modify_sample(input_sample)
        assert actual_res == correct_res
