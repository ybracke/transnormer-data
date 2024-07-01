import csv
import os

from typing import Dict, List, Iterable, Optional, Set, Tuple, Union

import datasets
import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer
from transnormer_data import utils


def find_sublist_indexes(main_list, sublist):
    """
    Helper function for get_index_map
    """
    sublist_length = len(sublist)
    main_length = len(main_list)

    for i in range(main_length - sublist_length + 1):
        if main_list[i : i + sublist_length] == sublist:
            return list(range(i, i + sublist_length))
    return None


def get_index_map(
    search_tuples: List[Tuple[int, ...]],
    src_indices: Iterable[int],
    trg_indices: List[int],
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """ """
    index_map = {}
    for search_tuple in search_tuples:
        sublist_indices = find_sublist_indexes(src_indices, search_tuple)
        if sublist_indices is None:
            continue
        # assert are_consecutive(positions_trg_indices)
        l = sublist_indices[0]
        u = sublist_indices[-1] + 1
        # convert list to a tuple without duplicates
        tuple_trg_indices = tuple(sorted(set(trg_indices[l:u])))
        index_map[search_tuple] = tuple_trg_indices
    return index_map


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

    def _src_mapping_to_trg_mapping(
        self,
        index_mapping: Dict[Tuple[int, ...], Tuple[int, ...]],
        ngrams2indices_src: Dict[Tuple[str, ...], List[Tuple[int, ...]]],
        repl_lex: Dict[Tuple[str, ...], Tuple[str, ...]],
    ) -> Dict[Tuple[str, ...], List[Tuple[int, ... ]]]:
        """ """
        target_ngrams2indices = {}
        for ngram_src, indices_src in ngrams2indices_src.items():
            ngram_trg = repl_lex.get(ngram_src)
            indices_trg = [index_mapping.get(i) for i in indices_src]
            if ngram_trg is None or None in indices_trg:
                continue
            target_ngrams2indices[ngram_trg] = indices_trg

        return target_ngrams2indices

    def _get_target_ngrams_and_indices(
        self,
        ngrams2indices_src: Dict[Tuple[str, ...], List[Tuple[int, ...]]],
        alignment: List[List[int]],
        replacement_lex: Dict[Tuple[str, ...], Tuple[str, ...]],
    ) -> Dict[Tuple[str, ...], List[Tuple[int, ...]]]:
        """ """
        src_indices, trg_indices = zip(*alignment)
        # flatten
        search_src_indices = [val for l in ngrams2indices_src.values() for val in l]
        index_map = get_index_map(search_src_indices, src_indices, trg_indices) # type: ignore
        return self._src_mapping_to_trg_mapping(index_map, ngrams2indices_src, replacement_lex)
