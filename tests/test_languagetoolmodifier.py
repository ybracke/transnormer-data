import csv
import pytest
import unittest

import datasets

from transnormer_data.modifier.language_tool_modifier import LanguageToolModifier


class LanguageToolModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.rule_file = "tests/testdata/languagetool/rules.txt"
        self.modifier = LanguageToolModifier(self.rule_file)

    def tearDown(self) -> None:
        pass

    def test_modify_sample(self) -> None:
        input_sample = {
            "norm": "Zuviel Abfluß.",
            "norm_tok": ["Zuviel", "Abfluß", "."],
            "norm_ws": [False, True, False],
            "norm_spans": [
                [0, 6],
                [7, 13],
                [13, 14],
            ],
            "orig_tok": ["Zuviel", "Abfluß", "."],
            "alignment": [[0, 0], [1, 1], [2, 2]],
        }
        target = {
            "norm": "Zu viel Abfluss.",
            "norm_tok": ["Zu", "viel", "Abfluss", "."],
            "norm_ws": [False, True, True, False],
            "norm_spans": [
                [0, 2],
                [3, 7],
                [8, 15],
                [15, 16],
            ],
            "orig_tok": ["Zuviel", "Abfluß", "."],
            "alignment": [[0, 0], [0, 1], [1, 2], [2, 3]],
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target
