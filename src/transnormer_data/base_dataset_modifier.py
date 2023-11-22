from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import spacy

from textalign import Aligner

MODEL = "de_dep_news_trf"  # TODO: not in base class


class BaseDatasetModifier:
    """Base class for implementation of modifiers"""

    def __init__(
        self, dataset: Optional[datasets.Dataset] = None, nlp_model=MODEL
    ) -> None:
        self.dataset = dataset
        self.modify_functions: Dict[Callable, dict]
        self.nlp = spacy.load(nlp_model)  # TODO: not in base class

    # def modify_dataset(self) -> None:
    #     """Apply the specified modification(s) to the entire dataset"""
    #     for func, args in self.modify_functions:
    #         self.dataset = self.dataset.map(
    #             self.modify_sample(func), fn_kwargs=args, batched=False
    #         )

    def update_tok_from_raw(self, sample: Dict, key_raw: str, key_tok: str, key_ws: str) -> Dict:
        """Update a sample's tokenized and whitespace entries based on its raw string entry"""
        sample[key_raw] = self._tok2raw(sample[key_tok], sample[key_ws])
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

    def update_raw_from_tok(self, sample: Dict, key_raw: str, key_tok: str, key_ws: str) -> Dict:
        """Update a sample's raw string entry based on its tokenized + whitespace entry"""
        sample[key_raw] = self._tok2raw(sample[key_tok], sample[key_ws])
        return sample

    def _tok2raw(self, tokens: List[str], whitespaces: Optional[List[bool]]) -> str:
        """Internal detokenization function"""
        if whitespaces is not None:
            raw = ""
            for ws, tok in zip(whitespaces, tokens):
                sep = " " if ws else ""
                raw += f"{sep}{tok}"
        # else:
        #     raw = self.detokenizer.detokenize(sample.tok)
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
        aligner.get_bidirectional_alignments()
        # Convert format of alignments from AlignedPairs to python list
        alignment = [list(pair) for pair in aligner.aligned_tokidxs]
        return alignment

    def update_spans(self):
        pass