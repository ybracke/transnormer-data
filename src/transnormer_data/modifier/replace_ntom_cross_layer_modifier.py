import csv
import os

from typing import Dict, List, Optional, Tuple, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer
from transnormer_data import utils


class ReplaceNtoMCrossLayerModifier(BaseDatasetModifier):
    def __init__(self, source_layer: str = "orig", target_layer: str = "norm", mapping_files: List[str] = []) -> None:
        """
        n-gram to m-gram cross layer replacement modifier.

        This modifier replaces token sequences in the tokenized target text (here "norm_tok" or "orig_tok") if this token sequence corresponds to a given sequence on the source layer. The changes are propagated to the raw version ("norm" or "orig"), new alignments are computed, etc.

        Example: All occurrences of the n-gram (X,Y) on the orig layer will 
        get normalized as (X',Y'). That is, we exchange the m-gram on the norm layer 
        that corresponds to (X,Y) with (X', Y')
        """
        pass
