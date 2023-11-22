from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import spacy


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

    def raw2tok(self, sample: Dict, key_raw: str, key_tok: str, key_ws: str) -> Dict:
        sample[key_raw] = self._tok2raw(sample[key_tok], sample[key_ws])
        return sample

    def _raw2tok(self, raw: str) -> Tuple[List[str], List[bool]]:
        """Internal tokenization Callable"""
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

    def tok2raw(self, sample: Dict, key_raw: str, key_tok: str, key_ws: str) -> Dict:
        sample[key_raw] = self._tok2raw(sample[key_tok], sample[key_ws])
        return sample

    def _tok2raw(self, tokens: List[str], whitespaces: List[bool]) -> str:
        """Internal detokenization function"""
        if whitespaces is not None:
            raw = ""
            for ws, tok in zip(whitespaces, tokens):
                sep = " " if ws else ""
                raw += f"{sep}{tok}"
        # else:
        #     raw = self.detokenizer.detokenize(sample.tok)
        return raw

    def update_spans(self):
        pass

    def update_alignment(self):
        pass
