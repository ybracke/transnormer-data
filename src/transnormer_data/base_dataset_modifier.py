import os
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import datasets
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textalign import Aligner

from transnormer_data import utils

MODEL = "de_core_news_sm"  # alternative, bigger model: "de_dep_news_trf"


class BaseDatasetModifier:
    """Base class for implementation of modifiers"""

    def __init__(self, spacy_model: str = MODEL) -> None:
        self.nlp = spacy.load(spacy_model)
        self.detokenizer: Optional[TreebankWordDetokenizer] = None

    def update_tok_from_raw(
        self, sample: Dict, key_raw: str, key_tok: str, key_ws: str
    ) -> Dict:
        """Update a sample's tokenized and whitespace entries based on its raw string entry"""
        tok, ws = self._raw2tok(sample[key_raw])
        sample[key_tok] = tok
        sample[key_ws] = ws
        return sample

    def _raw2tok(self, raw: str) -> Tuple[List[str], List[bool]]:
        """Internal tokenization function"""
        doc = self.nlp(raw.strip())
        tokens = []
        whitespaces = [
            False,
        ]
        for token in doc:
            tokens.append(token.text)
            ws = bool(len(token.whitespace_))  # False if length 0
            whitespaces.append(ws)
            # , token.idx)
        # pop final whitespace
        return tokens, whitespaces[:-1]

    def update_raw_from_tok(
        self, sample: Dict, key_raw: str, key_tok: str, key_ws: Optional[str] = None
    ) -> Dict:
        """Update a sample's raw string entry based on its tokenized (+ optionally whitespace) entry"""
        sample[key_raw] = self._tok2raw(sample[key_tok], sample.get(key_ws))
        return sample

    def _tok2raw(self, tokens: List[str], whitespaces: Optional[List[bool]]) -> str:
        """Internal detokenization function"""
        if whitespaces is not None:
            raw = ""
            for ws, tok in zip(whitespaces, tokens):
                sep = " " if ws else ""
                raw += f"{sep}{tok}"
        else:
            if self.detokenizer is None:
                raise Exception(
                    "Error while detokenizing: No whitespace information and no detokenizer."
                )
            raw = self.detokenizer.detokenize(tokens)
        return raw

    def update_alignment(
        self, sample: Dict, key_tokens_src: str, key_tokens_trg: str, key_alignment: str
    ) -> Dict:
        """Align the tokens from source and target and update the sample's alignment property"""
        alignment = self._align(sample[key_tokens_src], sample[key_tokens_trg])
        sample[key_alignment] = alignment
        return sample

    def _align(self, tokens_src: List[str], tokens_trg: List[str]) -> List[List[int]]:
        """Align the tokens from source and target"""
        aligner = Aligner(tokens_src, tokens_trg)
        # Set to compute maximum 1:4/4:1-alignments
        aligner.get_bidirectional_alignments(max_aligned_tokens=4)
        # Convert format of alignments from AlignedPairs to python list
        alignment = [list(pair) for pair in aligner.aligned_tokidxs]
        return alignment

    def update_token_spans(
        self, sample: Dict, key_tokens: str, key_ws: str, key_spans: str
    ) -> Dict:
        """Update the token spans from tokens and whitespace"""
        spans = self._get_token_spans(sample[key_tokens], sample[key_ws])
        sample[key_spans] = spans
        return sample

    def _get_token_spans(
        self, tokens: List[str], whitepaces: List[bool]
    ) -> List[List[int]]:
        """Internal function that calculates the token spans from ws and tokens"""
        spans = []
        start_idx = 0
        end_idx = 0
        for tok, ws in zip(tokens, whitepaces):
            start_idx = end_idx + bool(ws)  # add 1 if preceded by ws
            end_idx = start_idx + len(tok)
            spans.append([start_idx, end_idx])
        return spans

    def update_spans_and_ws_from_tok_and_raw(
        self, sample: Dict, key_tokens: str, key_raw: str, key_spans: str, key_ws: str
    ) -> Dict:
        """Update token spans and whitespaces from tokens and raw string"""
        spans, ws = self._get_spans_and_ws_from_tok_and_raw(
            sample[key_tokens], sample[key_raw]
        )
        sample[key_spans] = spans
        sample[key_ws] = ws
        return sample

    def _get_spans_and_ws_from_tok_and_raw(
        self, tokens: List[str], raw: str
    ) -> Tuple[List[List[int]], List[bool]]:
        """Calculate token spans and whitespaces from tokens and raw string"""
        start_idx = 0
        end_idx = 0
        whitespaces = []
        spans = []
        len_raw = len(raw)
        for tok in tokens:
            if end_idx >= len_raw:
                break
            # is the next token a space character?
            has_preceding_ws = raw[end_idx] == " "

            start_idx = end_idx + bool(has_preceding_ws)  # add 1 if preceded by ws
            end_idx = start_idx + len(tok)
            whitespaces.append(has_preceding_ws)
            spans.append([start_idx, end_idx])
        assert self._tok2raw(tokens, whitespaces) == raw
        return spans, whitespaces

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

    def get_idx2idxs(
        self, alignment: List[List[int | None]]
    ) -> Dict[int | None, List[int | None]]:
        """
        Create a mapping from every single src element in `alignment`
        to all trg elements in that src is associated to.

        Example output: {0 : [0,1], 1 : [2,]}
        """

        idx2idxs = {}
        for idx_src, idx_trg in alignment:
            if idx_src not in idx2idxs:
                idx2idxs[idx_src] = [idx_trg]
            else:
                idx2idxs[idx_src].append(idx_trg)
        return idx2idxs

    @abstractmethod
    def modify_sample(self, sample: Dict):
        pass
