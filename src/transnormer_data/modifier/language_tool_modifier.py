import os

from typing import Dict, Optional, Set, Union

import datasets
import spacy
from language_tool_python import LanguageTool

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data import utils


class LanguageToolModifier(BaseDatasetModifier):
    def __init__(self, rule_file: str) -> None:
        """
        This modifier applies the LanguageTool to the raw version of the target layer
        and propagates the changes to the tokenized version.

        Target layer is fixed to "norm", source layer is fixed to "orig".
        """

        # Keys in the sample dictionary
        self.raw = "norm"
        self.tok = "norm_tok"
        self.ws = "norm_ws"
        self.spans = "norm_spans"
        self.tok_src = "orig_tok"
        self.alignment = "alignment"

        # NLP
        self.nlp = spacy.blank("de")

        # LanguageTool instance
        self.langtool: LanguageTool = LanguageTool(
            language="de-DE", language_tool_download_version="6.3"
        )
        self.set_langtool_rules(self._load_rules(rule_file))

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

        Here, the modification is applied to norm_raw (via LanguageTool) and
        the changes are propagated to norm_tok, etc.
        """

        # Update raw via LanguageTool
        raw_old = sample[self.raw]
        raw_new = self.langtool.correct(raw_old)
        any_changes = raw_new != raw_old
        if any_changes:
            sample[self.raw] = raw_new
            self.update_tok_from_raw(
                sample, key_raw=self.raw, key_tok=self.tok, key_ws=self.ws
            )

            # Update spans
            self.update_spans_and_ws_from_tok_and_raw(
                sample,
                key_tokens=self.tok,
                key_raw=self.raw,
                key_ws=self.ws,
                key_spans=self.spans,
            )

            # Update alignment
            self.update_alignment(
                sample,
                key_tokens_src=self.tok_src,
                key_tokens_trg=self.tok,
                key_alignment=self.alignment,
            )
        return sample

    def _load_rules(self, file: str) -> Set[str]:
        """
        Load the file with the LanguageTool rule identifiers as a set of strings

        `file` must be a path to a text file that contains a list of identifiers for LanguageTool rules (e.g. OLD_SPELLING), one rule ID per line.
        """
        rules_set = set()
        with open(file, "r") as f:
            for line in f:
                rule_id = line.strip()  # Remove leading/trailing whitespace
                if rule_id:  # Skip empty lines
                    rules_set.add(rule_id)
        return rules_set

    def set_langtool_rules(self, rules: Set[str]) -> None:
        """
        Set which rules the LanguageTool should apply. It will not apply
        any other rules. Rules must be given by their rule ID, which is defined
        by LanguageTool.
        """
        # TODO: Check if rules actually exist
        self.langtool.enabled_rules = rules
        self.langtool.enabled_rules_only = True
