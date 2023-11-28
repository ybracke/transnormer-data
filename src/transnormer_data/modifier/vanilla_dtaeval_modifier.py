from typing import Any, Dict

import datasets

from transnormer_data.base_dataset_modifier import BaseDatasetModifier

from nltk.tokenize.treebank import TreebankWordDetokenizer



class VanillaDtaEvalModifier(BaseDatasetModifier):
    def __init__(self, dataset: datasets.Dataset) -> None:
        """
        Modifier for the DtaEvalMaker

        This modifier simply calls the functions that compute raw versions, whitespace and alignments from the tokenized version.

        """
        # Dataset
        self.dataset = dataset

        # Keys for the relevant properties
        self.key_src_raw = "orig"
        self.key_src_tok = "orig_tok"
        self.key_src_ws = "orig_ws"

        self.key_trg_raw = "norm"
        self.key_trg_tok = "norm_tok"
        self.key_trg_ws = "norm_ws"

        self.key_alignment = "alignment"

        # Detokenizer
        self.detokenizer = TreebankWordDetokenizer()

    def modify_dataset(self) -> datasets.Dataset:
        self.dataset = self.dataset.map(self.modify_sample)
        return self.dataset

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        Here, we have no modifications and simply call the raw-stringification function

        """
        self.update_raw_from_tok(sample, key_raw=self.key_src_raw, key_tok=self.key_src_tok, key_ws=self.key_src_ws)
        self.update_raw_from_tok(sample, key_raw=self.key_trg_raw, key_tok=self.key_trg_tok, key_ws=self.key_trg_ws)

        return sample
