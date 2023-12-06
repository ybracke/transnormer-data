from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import spacy

from textalign import Aligner

MODEL = "de_dep_news_trf"  # TODO: not in base class
MODEL = "de_core_news_sm"

class BaseDatasetModifier:
    """Base class for implementation of modifiers"""

    def __init__(
        self, dataset: Optional[datasets.Dataset] = None, nlp_model=MODEL
    ) -> None:
        self.dataset = dataset
        self.modify_functions: Dict[Callable, dict]
        self.nlp = spacy.load(nlp_model)  # TODO: not in base class
        self.detokenizer = None

    # def modify_dataset(self) -> None:
    #     """Apply the specified modification(s) to the entire dataset"""
    #     for func, args in self.modify_functions:
    #         self.dataset = self.dataset.map(
    #             self.modify_sample(func), fn_kwargs=args, batched=False
    #         )

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
        self, sample: Dict, key_raw: str, key_tok: str, key_ws: Optional[str]
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
                raise("Error while detokenizing: No whitespace information and no detokenizer.")
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
        spans, ws = self._get_spans_and_ws_from_tok_and_raw(sample[key_tokens], sample[key_raw])
        sample[key_spans] = spans
        sample[key_ws] = ws
        return sample

    def _get_spans_and_ws_from_tok_and_raw(
            self, tokens: List[str], raw:str
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
            has_preceding_ws = (raw[end_idx] == " ")

            start_idx = end_idx + bool(has_preceding_ws)  # add 1 if preceded by ws
            end_idx = start_idx + len(tok)
            whitespaces.append(has_preceding_ws)
            spans.append([start_idx, end_idx])
        assert self._tok2raw(tokens, whitespaces) == raw
        return spans, whitespaces