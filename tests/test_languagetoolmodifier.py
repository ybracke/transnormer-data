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

    def test_modify_dataset(self) -> None:
        data_files = ["tests/testdata/jsonl/dtak/varnhagen_rahel01_1834.jsonl"]
        dataset = datasets.load_dataset("json", data_files=data_files, split="train")
        dataset_mod = self.modifier.modify_dataset(dataset)
        assert (
            dataset[20]["norm"]
            == "Und wenn auch der volle Reichtum dieses von Geist und Liebe beseelten Gemütes nicht unmittelbar jedem Auge ganz entfaltet lag, so bekennen doch Alle, die auch nur Momente dieses in Wohlwollen und Wahrheitseifer stets erregten Lebens angeschaut, daß sie von dieser Erscheinung einen seltenen und ahndungsvollen Eindruck der eigentümlichsten Kraft und Anmut empfangen haben, der jeder freigebigsten Voraussetzung Raum gibt, und Alle mitfühlend unserer Wehklage beistimmen läßt."
        )
        assert (
            dataset_mod[20]["norm"]
            == "Und wenn auch der volle Reichtum dieses von Geist und Liebe beseelten Gemütes nicht unmittelbar jedem Auge ganz entfaltet lag, so bekennen doch Alle, die auch nur Momente dieses in Wohlwollen und Wahrheitseifer stets erregten Lebens angeschaut, daß sie von dieser Erscheinung einen seltenen und ahndungsvollen Eindruck der eigentümlichsten Kraft und Anmut empfangen haben, der jeder freigebigsten Voraussetzung Raum gibt, und Alle mitfühlend unserer Wehklage beistimmen lässt."
        )
