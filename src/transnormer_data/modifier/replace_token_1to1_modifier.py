import csv
import os

from typing import Dict, List, Optional, Tuple, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer
from transnormer_data import utils


class ReplaceToken1to1Modifier(BaseDatasetModifier):
    def __init__(
        self, layer: str = "norm", mapping_files: Optional[List[str]] = None
    ) -> None:
        """
        Example implementation of a type replacement modifier

        This modifier replaces types on the tokenized version of the target layer
        (here "norm_tok") and propagates the changes to the raw version ("norm")

        """

        # Keys in the sample dictionary
        valid_layers = {"norm", "orig"}
        if layer not in valid_layers:
            raise ValueError(
                f"ReplaceToken1to1Modifier: layer must be one of{valid_layers}"
            )
        self.raw = f"{layer}"
        self.tok = f"{layer}_tok"
        self.ws = f"{layer}_ws"
        self.spans = f"{layer}_spans"

        # Detokenizer
        self.detokenizer = DtaEvalDetokenizer()

        # NLP
        self.nlp = spacy.blank("de")

        # Replacement dictionary
        mapping_files = [] if mapping_files is None else mapping_files
        self.type_mapping: Dict[str, str] = self._load_replacement_mapping(
            mapping_files
        )

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        E.g., if the modification was applied to norm_tok,
        the changes have to be propagated to norm_raw.
        """
        tokens_old = sample[self.tok]
        tokens_new, any_changes = self.map_tokens(tokens_old)
        sample[self.tok] = tokens_new
        if any_changes:
            self.update_raw_from_tok(
                sample, key_raw=self.raw, key_tok=self.tok, key_ws=self.ws
            )
            self.update_spans_and_ws_from_tok_and_raw(
                sample,
                key_tokens=self.tok,
                key_raw=self.raw,
                key_ws=self.ws,
                key_spans=self.spans,
            )

        return sample

    def map_tokens(self, tokens_old: List[str]) -> Tuple[List[str], bool]:
        """Modifies `tokens_old` by applying the type mapping on each token, if necessary. Returns a tuple `(tokens_new, any_changes)` where `any_changes` is False iff `tokens_new==tokens_old`."""
        tokens_new = []
        any_changes = False
        for t in tokens_old:
            token_new = self.type_mapping.get(t)
            # Found something?
            if token_new is not None:
                any_changes = True
                tokens_new.append(token_new)
            else:
                tokens_new.append(t)
        return tokens_new, any_changes

    def _load_replacement_mapping(self, files: List[str]) -> Dict[str, str]:
        all_pairs = []
        for file in files:
            with open(file, newline="") as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                dialect.quotechar = "`"  # FIXME
                csvfile.seek(0)
                reader = csv.reader(csvfile, dialect)
                for row in reader:
                    pair = tuple(row)
                    assert len(pair) == 2, print(pair)
                    all_pairs.append(pair)
        replacement_mapping: Dict[str, str] = dict(all_pairs)
        return replacement_mapping
