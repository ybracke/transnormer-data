import os
import unittest

import datasets
import pytest

from transnormer_data.modifier.language_detection_modifier import (
    LanguageDetectionModifier,
)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions"
)
class LanguageDetectionModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = LanguageDetectionModifier()

    def tearDown(self) -> None:
        pass

    def test_langdetec_simple(self) -> None:
        guesses = self.modifier.languagedetector(
            "Das ist ein einfach zu erkennender deutscher Satz."
        )
        assert guesses == {
            "lang_fastText": "de",
            "lang_py3langid": "de",
            "lang_cld3": "de",
        }

    def test_test_langdetec_latin(self) -> None:
        guesses = self.modifier.languagedetector("Homos homini lupus.")
        # Does not work so well
        assert guesses == {
            "lang_fastText": "en",
            "lang_py3langid": "es",
            "lang_cld3": "la",
        }

    def test_langdetec_oldgerman(self) -> None:
        guesses = self.modifier.languagedetector(
            "DVRchleuchtigſte/ Hochgeborne Churfuͤrſtin/ Gnaͤ-digſte Fraw/ meine Gottſelige liebe Mutter Anna Wecke- rin/ hat kurtz vor jrem ſeligen Ende gegenwertig Kochbuch/ ſo ſie auß langer eigner Vbung vnnd Erfahrung viel Jahr hero fleiſſig zuſammen getragen/ endlich/ auff embſiges an-halten viler guthertziger/ vnd der ſachen verſtaͤndiger lent/ zum Truck ver-fertiget/ vnd iſt ſolches E. C. G. zuzuſchreiben hoͤchlich begierig geweſen."
        )
        assert guesses == {
            "lang_fastText": "de",
            "lang_py3langid": "de",
            "lang_cld3": "de",
        }

    def test_modify_sample(self) -> None:
        input_sample = {
            "orig": "Zuviel Abfluß.",
            "norm": "Zu viel Abfluss.",
        }
        target = {
            "orig": "Zuviel Abfluß.",
            "norm": "Zu viel Abfluss.",
            "lang_fastText": "de",
            "lang_py3langid": "de",
            "lang_cld3": "de",
            "lang_de": 1.0,
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target

        input_sample = {
            "orig": "Donde Lebensmann.",
            "norm": "Donde Lebensmann.",
        }
        target = {
            "orig": "Donde Lebensmann.",
            "norm": "Donde Lebensmann.",
            "lang_fastText": "en",
            "lang_py3langid": "de",
            "lang_cld3": "hu",
            "lang_de": 0.333,
        }
        result = self.modifier.modify_sample(input_sample)
        assert result == target

    def test_modify_dataset(self) -> None:
        data_files = ["tests/testdata/jsonl/dtak/varnhagen_rahel01_1834.jsonl"]
        dataset = datasets.load_dataset("json", data_files=data_files, split="train")
        dataset_mod = self.modifier.modify_dataset(dataset, save_to=False)
        assert dataset_mod[20]["lang_fastText"] == "de"
        assert dataset_mod[20]["lang_py3langid"] == "de"
        assert dataset_mod[20]["lang_cld3"] == "de"
        assert dataset_mod[20]["lang_de"] == 1.0
