from typing import Any, Dict, List, Optional, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier, MODEL
from transnormer_data.detokenizer import DtaEvalDetokenizer


class VanillaDtaModifier(BaseDatasetModifier):
    def __init__(self) -> None:
        """
        Modifier for the DtaEvalMaker

        This modifier simply calls the functions that compute raw versions, whitespace and alignments from the tokenized version.

        """

        # Keys for the relevant properties
        self.key_src_raw = "orig"
        self.key_src_tok = "orig_tok"
        self.key_src_ws = "orig_ws"
        self.key_src_spans = "orig_spans"

        self.key_trg_raw = "norm"
        self.key_trg_tok = "norm_tok"
        self.key_trg_ws = "norm_ws"
        self.key_trg_spans = "norm_spans"

        self.key_alignment = "alignment"

        # Detokenizer
        self.detokenizer = DtaEvalDetokenizer()

        # NLP
        self.nlp = spacy.load(
            MODEL,
            disable=[
                "tok2vec",
                "tagger",
                "morphologizer",
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "ner",
            ],
        )

    def modify_dataset(self, dataset: datasets.Dataset, save_to = None) -> datasets.Dataset:
        dataset = dataset.map(self.modify_sample)
        return dataset

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        """

        # Detokenize tok to produce raw text version for source and target
        self.update_raw_from_tok(
            sample,
            key_raw=self.key_src_raw,
            key_tok=self.key_src_tok,
            key_ws=self.key_src_ws,
        )
        self.update_raw_from_tok(
            sample,
            key_raw=self.key_trg_raw,
            key_tok=self.key_trg_tok,
            key_ws=self.key_trg_ws,
        )

        # Tokenize target again. Reason: TODO
        self.update_tok_from_raw(
            sample,
            key_raw=self.key_trg_raw,
            key_tok=self.key_trg_tok,
            key_ws=self.key_trg_ws,
        )

        # Compute alignments
        self.update_alignment(
            sample,
            key_tokens_src=self.key_src_tok,
            key_tokens_trg=self.key_trg_tok,
            key_alignment=self.key_alignment,
        )

        # Compute spans
        self.update_spans_and_ws_from_tok_and_raw(
            sample,
            key_tokens=self.key_src_tok,
            key_raw=self.key_src_raw,
            key_spans=self.key_src_spans,
            key_ws=self.key_src_ws,
        )
        self.update_spans_and_ws_from_tok_and_raw(
            sample,
            key_tokens=self.key_trg_tok,
            key_raw=self.key_trg_raw,
            key_spans=self.key_trg_spans,
            key_ws=self.key_trg_ws,
        )

        return sample
