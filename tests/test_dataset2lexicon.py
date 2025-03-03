import json
import unittest
from typing import Any, Dict, List

from transnormer_data.cli import dataset2lexicon


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return data


class FooTester(unittest.TestCase):
    def setUp(self):
        self.data = read_jsonl("tests/testdata/testfile_002.jsonl")

    def test_transform_alignment(self):
        alignment = dataset2lexicon.transform_alignment(self.data[0]["alignment"])
        target_index_alignment = [
            ((0, 1), (0,)),
            ((2,), (1,)),
            ((3,), (2,)),
            ((4,), (3, 4)),
            ((5,), (5,)),
        ]
        assert alignment == target_index_alignment
        print(alignment)

        orig_tok = self.data[0]["orig_tok"]
        norm_tok = self.data[0]["norm_tok"]

        target_ngram_alignment = [
            ("Irgend eyn", "Irgendein"),
            ("ſchoenes", "schönes"),
            ("Haus", "Haus"),
            ("zuviel", "zu viel"),
            (".", "."),
        ]
        ngram_alignment = dataset2lexicon.get_ngram_alignment(
            alignment, orig_tok, norm_tok
        )

        assert ngram_alignment == target_ngram_alignment
