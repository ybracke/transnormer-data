import json
from typing import Any, Dict, List
import unittest

from transnormer_data.base_dataset_modifier import BaseDatasetModifier


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return data


class BasicsTester(unittest.TestCase):
    def setUp(self):
        self.data = read_jsonl("tests/testdata/testfile_001.jsonl")
        self.modifier = BaseDatasetModifier()

    def test_raw2tok(self):
        orig_tok_from_raw, orig_ws_from_raw = self.modifier._raw2tok(
            self.data[0]["orig"]
        )
        assert orig_tok_from_raw == self.data[0]["orig_tok"]
        assert orig_ws_from_raw == self.data[0]["orig_ws"]

        norm_tok_from_raw, norm_ws_from_raw = self.modifier._raw2tok(
            self.data[0]["norm"]
        )
        assert norm_tok_from_raw == self.data[0]["norm_tok"]
        assert norm_ws_from_raw == self.data[0]["norm_ws"]

    def test_tok2raw(self):
        orig_raw_from_tok = self.modifier._tok2raw(
            self.data[0]["orig_tok"], self.data[0]["orig_ws"]
        )
        assert orig_raw_from_tok == self.data[0]["orig"]

        norm_raw_from_tok = self.modifier._tok2raw(
            self.data[0]["norm_tok"], self.data[0]["norm_ws"]
        )
        assert norm_raw_from_tok == self.data[0]["norm"]

    def test_compute_spans(self):
        spans = self.modifier._get_token_spans(
            self.data[0]["orig_tok"], self.data[0]["orig_ws"]
        )
        # assert that the same spans where compute
        assert spans == self.data[0]["orig_spans"]

        # assert that you get the tokens when you extract the spans from raw
        tokens = self.data[0]["orig_tok"]
        raw = self.data[0]["orig"]
        for span, token in zip(spans, tokens):
            start, end = span[0], span[1]
            assert raw[start:end] == token

        # same for norm
        spans = self.modifier._get_token_spans(
            self.data[0]["norm_tok"], self.data[0]["norm_ws"]
        )
        assert spans == self.data[0]["norm_spans"]

        tokens = self.data[0]["norm_tok"]
        raw = self.data[0]["norm"]
        for span, token in zip(spans, tokens):
            start, end = span[0], span[1]
            assert raw[start:end] == token

    def test_interal_align(self):
        alignment = self.modifier._align(
            self.data[0]["orig_tok"], self.data[0]["norm_tok"]
        )
        assert alignment == self.data[0]["alignment"]
