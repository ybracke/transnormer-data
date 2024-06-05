import os
from typing import List
import unittest

import datasets

from transnormer_data.detokenizer import DtaEvalDetokenizer


class DtaEvalDetokenizerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.detokenizer = DtaEvalDetokenizer()

    def test_detokenize_01(self) -> None:
        tokens = ["Ich", "bin", "ein", "Mensch", "."]
        target_raw = "Ich bin ein Mensch."
        detokenized = self.detokenizer.detokenize(tokens)
        assert target_raw == detokenized

    def test_detokenize_02(self) -> None:
        tokens = [
            "„",
            "Irgend",
            "'",
            "was",
            ";",
            "es",
            "iſt",
            "ganz",
            "gleich",
            ",",
            "es",
            "muß",
            "nur",
            "einen",
            "Reim",
            "auf",
            "‚",
            "u",
            "‘",
            "haben",
            ";",
            "‚",
            "u",
            "‘",
            "iſt",
            "immer",
            "Trauervokal",
            ".",
        ]
        target_raw = "„Irgend' was; es iſt ganz gleich, es muß nur einen Reim auf ‚u‘ haben; ‚u‘ iſt immer Trauervokal."
        detokenized = self.detokenizer.detokenize(tokens)
        assert target_raw == detokenized
