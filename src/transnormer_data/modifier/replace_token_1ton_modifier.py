import csv
import os

from typing import Dict, List, Optional, Tuple, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer
from transnormer_data import utils


class ReplaceToken1toNModifier(BaseDatasetModifier):
    def __init__(self, layer: str = "norm", mapping_files: List[str] = []) -> None:
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
        other_layer = (valid_layers - {layer}).pop()
        self.tok_src = f"{other_layer}_tok"
        self.alignment = "alignment"

        # Detokenizer
        self.detokenizer = DtaEvalDetokenizer()

        # NLP
        self.nlp = spacy.blank("de")

        # Replacement dictionary
        self.type_mapping: Dict[str, List[str]] = self._load_replacement_mapping(
            mapping_files
        )

    def modify_dataset(
        self,
        dataset: datasets.Dataset,
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Union[datasets.Dataset, None]:
        dataset = dataset.map(self.modify_sample)
        if save_to:
            if not os.path.isdir(save_to):
                os.makedirs(save_to)
            utils.save_dataset_to_json_grouped_by_property(
                dataset, property="basename", path_outdir=save_to
            )
        return dataset

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        E.g., if the modification was applied to norm_tok,
        the changes have to be propagated to norm_raw.
        """
        tokens_old = sample[self.tok]
        ws_old = sample[self.ws]
        tokens_new, ws_new, any_changes = self.map_tokens(tokens_old, ws_old)
        if any_changes:
            sample[self.tok] = tokens_new
            sample[self.ws] = ws_new
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
            self.update_alignment(
                sample,
                key_tokens_src=self.tok_src,
                key_tokens_trg=self.tok,
                key_alignment=self.alignment,
            )

        return sample

    def map_tokens(
        self, tokens_old: List[str], ws_old: List[bool]
    ) -> Tuple[List[str], List[bool], bool]:
        """Modifies `tokens_old` and `ws_old` by applying the type mapping on each token, if necessary. Returns a tuple `(tokens_new, any_changes)` where `any_changes` is False iff `tokens_new==tokens_old`."""
        tokens_new = []
        ws_new = []
        any_changes = False
        for t, ws in zip(tokens_old, ws_old):
            _tokens_new = self.type_mapping.get(t)
            # Found something?
            if _tokens_new is not None:
                any_changes = True
                tokens_new.extend(_tokens_new)
                # ws := is the current token preceded by whitespace
                # so we keep the ws value of the original token in the beginning and add whitespace in front of the following tokens
                ws_new.extend([ws] + [True for i in range(len(_tokens_new) - 1)])
            else:
                tokens_new.append(t)
                ws_new.append(ws)
        return tokens_new, ws_new, any_changes

    def _load_replacement_mapping(self, files: List[str]) -> Dict[str, List[str]]:
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
                    mapto = pair[1].split(" ")
                    all_pairs.append((pair[0], mapto))
        replacement_mapping: Dict[str, List[str]] = dict(all_pairs)
        return replacement_mapping
