from typing import Any, Dict

import datasets

from transnormer_data.base_dataset_modifier import BaseDatasetModifier


class TypeReplacementModifier(BaseDatasetModifier):
    def __init__(self, dataset: datasets.Dataset) -> None:
        """
        Example implementation of a type replacement modifier

        This modifier replaces types on the tokenized version of the target layer
        (here "norm_tok") and propagates the changes to the raw version ("norm")

        """
        # Dataset
        self.dataset = dataset
        # Replacement dictionary
        self.type_mapping: Dict[str, str] = {"schönes": "tolles"}

        # Keys in the the sample dictionary
        self.raw = "norm"
        self.tok = "norm_tok"
        self.ws = "norm_ws"

        # Detokenizer
        # TODO

    def modify_dataset(self) -> datasets.Dataset:
        self.dataset = self.dataset.map(self.modify_sample)
        return self.dataset

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        E.g., if the modification was applied to norm_tok,
        the changes have to be propagated to norm_raw.
        """
        tokens_old = sample[self.tok]

        # Actual modification function
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

        sample[self.tok] = tokens_new
        if any_changes:
            self.tok2raw(sample, key_raw=self.raw, key_tok=self.tok, key_ws=self.ws)

            # Not necessary here, iff we only do 1:1 token replacements
            # If token replacement can be n:1 token replacements (and vice versa),
            #  e.g. 'zu mindest -> zumindest' we have to compute new alignments
            self.update_alignment(
                sample,
                key_tokens_src="orig_tok",
                key_tokens_trg="norm_tok",
                key_alignment="alignment",
            )

        return sample
