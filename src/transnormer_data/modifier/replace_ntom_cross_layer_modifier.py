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

    def _get_index_map(
        self,
        search_seqs: List[Iterable[int]],
        alignment: List[List[int]],
    ) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """
        Creates a mapping of indexes: src -> trg

        Given a list of sub-sequences from source (search_seqs) and an alignment,
        returns the mapping of of source sequences to the corresponding target sequences.

        Helper function for _get_idx2ngram_trg
        """

        idx_src2idxs_trg = self.get_idx2idxs(alignment)
        index_map = {}
        # Iterate over index sequences from source
        for search_seq in search_seqs:
            # Collect target indices
            indices_all_trg = []
            # Look at every element of src index sequence
            for idx_src in search_seq:
                # get all matching trg indices and add to collection
                idxs_trg = idx_src2idxs_trg.get(idx_src, [])
                indices_all_trg.extend(idxs_trg)
            # put into mapping
            indices_src = tuple(search_seq)
            # sort and remove dublicates from indices_all_trg
            indices_all_trg = sorted(set(indices_all_trg))
            index_map[indices_src] = tuple(indices_all_trg)

        return index_map

    def _get_idx2ngram_trg(
        self,
        ngrams2indices_src: Dict[Tuple[str, ...], List[Tuple[int, ...]]],
        alignment: List[List[int]],
        repl_lex: Dict[Tuple[str, ...], Tuple[str, ...]],
    ) -> Dict[Tuple[int, ...], Tuple[str, ...]]:
        """
        Create a mapping of a source index tuple to a desired source ngram according to
        replacement lexicon

        """
        if not (len(alignment)):
            return {}
        # flatten
        search_src_indices = [val for l in ngrams2indices_src.values() for val in l]
        # map source to target indices
        index_mapping = self._get_index_map(search_src_indices, alignment)  # type: ignore
        # create a mapping of a source index tuple to a source ngram
        # {(int, ...) : (str, ...)}
        idx2ngram_trg = {}
        for ngram_src, indices_src in ngrams2indices_src.items():
            # src_ngram in replacement lex?
            ngram_trg = repl_lex.get(ngram_src)
            indices_trg = [index_mapping.get(i) for i in indices_src]
            for idx_trg in indices_trg:
                if idx_trg is not None and ngram_trg is not None:
                    idx2ngram_trg[idx_trg] = ngram_trg
        return idx2ngram_trg

    def _get_start2ngram_and_end(
        self,
        idx2ngram: Dict[Tuple[int, ...], Tuple[str, ...]],
        remove_overlap: bool = True,
    ) -> Dict[int, Tuple[Tuple[str, ...], int]]:
        """
        Converts the output of _get_idx2ngram_trg to the form: {start : (ngram, end)}

        Optional: If spans of indices overlap only keep the first one
        """
        start2ngram_and_end = {}
        prev_end = -1
        for idxs, ngram in idx2ngram.items():
            start = idxs[0]
            end = idxs[-1]
            if remove_overlap and (start <= prev_end):
                # TODO: log; return identifiers (basename, sent_id)
                print(
                    f"Dropped ngram at position [{start} : {end}] because of overlap with previous ngram."
                )
                continue
            else:
                start2ngram_and_end[start] = (ngram, end)
                prev_end = end
        return start2ngram_and_end

    def _update_target_tok(
        self,
        target_tok_in: List[str],
        idx2ngram: Dict[Tuple[int, ...], Tuple[str, ...]],
        remove_overlap: bool = True,
    ):
        """ 
        Updates a token sequence according to the positions and changes 
        specified in idx2ngram
        """
        target_tok_out = []
        i = 0
        start2ngram_and_end = self._get_start2ngram_and_end(idx2ngram, remove_overlap)
        while i < len(target_tok_in):
            if i in start2ngram_and_end:
                ngram, end = start2ngram_and_end[i]
                target_tok_out += list(ngram)
                # jump forward
                i = end + 1
            else:
                target_tok_out.append(target_tok_in[i])
                i += 1
        return target_tok_out
