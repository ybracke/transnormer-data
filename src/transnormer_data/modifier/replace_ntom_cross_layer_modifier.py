import csv
import os

from typing import Dict, List, Iterable, Optional, Set, Tuple, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer
from transnormer_data import utils


class ReplaceNtoMCrossLayerModifier(BaseDatasetModifier):
    def __init__(
        self,
        source_layer: str = "orig",
        target_layer: str = "norm",
        mapping_files: List[str] = [],
    ) -> None:
        """
        n-gram to m-gram cross layer replacement modifier.

        This modifier replaces token sequences in the tokenized target text (here "norm_tok" or "orig_tok") if this token sequence corresponds to a given sequence on the source layer. The changes are propagated to the raw version ("norm" or "orig"), new alignments are computed, etc.

        Example: All occurrences of the n-gram (X,Y) on the orig layer will
        get normalized as (X',Y'). That is, we exchange the m-gram on the norm layer
        that corresponds to (X,Y) with (X', Y')
        """

    def _find_ngram_indices(
        self,
        ngrams_to_search: Set[Tuple[str]],
        sent_tok: List[str],
        ngram_lengths: Optional[Iterable[int]] = None,
    ) -> Dict[Tuple[str, ...], List[Tuple[int, ...]]]:
        """
        Returns a mapping of the given ngrams to their indices in the list of strings.
        """

        # Get ngram length's if not given
        if ngram_lengths is None:
            ngram_lengths = {len(ngram) for ngram in ngrams_to_search}
        ngram_lengths = sorted(ngram_lengths)

        # Partition sent_tok into lists of kgrams
        partitions = {}
        for n in ngram_lengths:
            partitions[n] = [
                tuple(sent_tok[i : i + n])  # noqa: E203
                for i in range(0, len(sent_tok))
                if len(sent_tok[i : i + n]) == n  # noqa: E203
            ]

        # Mapping to return, will be e.g. {('x','y') : [(1,2), (3,4)], ...}
        ngram2indices: Dict[Tuple[str, ...], List[Tuple[int, ...]]] = {}
        # Iterate over each partition and check for ngram
        for k in ngram_lengths:
            # Partition for kgrams
            for i, kgram in enumerate(partitions[k]):
                if kgram in ngrams_to_search:
                    indices = tuple(range(i, i + k))
                    if kgram in ngram2indices:
                        ngram2indices[kgram].append(indices)
                    else:
                        ngram2indices[kgram] = [indices]

        return ngram2indices
