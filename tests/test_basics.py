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


def _tok2raw(tokens: List[str], whitespaces: List[bool]):
    if whitespaces is not None:
        raw = ""
        for ws, tok in zip(whitespaces, tokens):
            sep = " " if ws else ""
            raw += f"{sep}{tok}"
    return raw


def dummy_modifier(text: str) -> str:
    return text


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
        orig_raw_from_tok = _tok2raw(self.data[0]["orig_tok"], self.data[0]["orig_ws"])
        assert orig_raw_from_tok == self.data[0]["orig"]

        norm_raw_from_tok = _tok2raw(self.data[0]["norm_tok"], self.data[0]["norm_ws"])
        assert norm_raw_from_tok == self.data[0]["norm"]

    def test_extract_spans(self):
        spans = get_spans(
            self.data[0]["orig"], self.data[0]["orig_tok"], self.data[0]["orig_ws"]
        )
        assert spans == self.data[0]["orig_spans"]

        spans = get_spans(
            self.data[0]["norm"], self.data[0]["norm_tok"], self.data[0]["norm_ws"]
        )
        assert spans == self.data[0]["norm_spans"]

    def test_interal_align(self):
        alignment = self.modifier._align(
            self.data[0]["orig_tok"], self.data[0]["norm_tok"]
        )
        assert alignment == self.data[0]["alignment"]
