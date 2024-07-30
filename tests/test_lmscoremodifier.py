import os
import pytest
import unittest

from transnormer_data.modifier.lm_score_modifier import LMScoreModifier, LMScorer


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions"
)
class LMScoreModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.lm_scorer = LMScorer(model_name="dbmdz/german-gpt2")
        self.modifier = LMScoreModifier()

    def tearDown(self) -> None:
        pass

    def test_scorer(self) -> None:
        scores_01 = self.lm_scorer("Das ist ein normaler deutscher Satz.")
        scores_02 = self.lm_scorer("normaler Satz deutscher Das ist nicht ein.")
        # higher is better
        assert scores_01["dbmdz/german-gpt2"] > scores_02["dbmdz/german-gpt2"]

    def test_modifier(self) -> None:
        sample_01 = {"norm": "Das ist ein normaler deutscher Satz."}
        sample_02 = {"norm": "normaler Satz deutscher Das ist nicht ein."}
        sample_01 = self.modifier.modify_sample(sample_01)
        sample_02 = self.modifier.modify_sample(sample_02)
        # lower is better
        assert sample_01["dbmdz/german-gpt2"] < sample_02["dbmdz/german-gpt2"]
