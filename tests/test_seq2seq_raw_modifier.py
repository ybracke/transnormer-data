import os
import pytest
import unittest

from transnormer_data.modifier.seq2seq_raw_modifier import (
    Seq2SeqRawModifier,
)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions"
)
class Seq2SeqRawModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        model = "ybracke/transnormer-dtaeval-v01"
        tokenizer = "ybracke/transnormer-dtaeval-v01"
        self.modifier = Seq2SeqRawModifier(model, tokenizer)

    def tearDown(self) -> None:
        pass

    def test_nothing(self) -> None:
        print(type(self.modifier.model))
