import csv
import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer

logger = logging.getLogger(__name__)


class ReplaceNtoMCrossLayerModifier(BaseDatasetModifier):
    def __init__(
        self,
        source_layer: str = "orig",
        target_layer: str = "norm",
        mapping_files: Optional[List[str]] = None,
        mapping_files_delimiters: Optional[str] = None,
    ) -> None:
        """
        n-gram to m-gram cross layer replacement modifier.

        This modifier replaces token sequences in the tokenized target text (here "norm_tok" or "orig_tok") if this token sequence corresponds to a given sequence on the source layer. The changes are propagated to the raw version ("norm" or "orig"), new alignments are computed, etc.

        Example: All occurrences of the n-gram (X,Y) on the orig layer will
        get normalized as (X',Y'). That is, we exchange the m-gram on the norm layer
        that corresponds to (X,Y) with (X', Y')
        """

        # Keys in the sample dictionary
        self.raw_trg = f"{target_layer}"
        self.tok_trg = f"{target_layer}_tok"
        self.ws_trg = f"{target_layer}_ws"
        self.spans_trg = f"{target_layer}_spans"
        self.tok_src = f"{source_layer}_tok"
        self.alignment = "alignment"
        # Score in range {0, 1} that tells us the proportion of language
        # detection models that guessed that the sample is in German
        self.lang_de_score = "lang_de"

        # Detokenizer
        self.detokenizer = DtaEvalDetokenizer()

        # Lengths of the source ngrams in the mapping
        self.src_ngram_lengths: List[int] | None = None

        # Replacement dictionary
        mapping_files = [] if mapping_files is None else mapping_files
        self.replacement_mapping: Dict[
            Tuple[str, ...], Tuple[str, ...]
        ] = self._load_n2m_replacement_mapping(mapping_files, mapping_files_delimiters)

        self._current_sample: Dict = {}

    def get_ngram_lengths(
        self, ngram_mapping: Dict[Tuple[str, ...], Tuple[str, ...]]
    ) -> List[int]:
        return sorted({len(ngram) for ngram in ngram_mapping.keys()})

    def _find_ngram_indices(
        self,
        ngrams_to_search: Set[Tuple[str, ...]],
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
        alignment: List[List[int | None]],
    ) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """
        Creates a mapping of indexes: src -> trg

        Given a list of sub-sequences from source (search_seqs) and an alignment,
        returns the mapping of of source sequences to the corresponding target sequences. 'None' elements are removed from trg.

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
            # Important: remove None elements from trg indices
            indices_all_trg_cleaned = sorted(
                set([i for i in indices_all_trg if i is not None])
            )
            # Make sure tuple is not empty
            if indices_all_trg_cleaned:
                index_map[indices_src] = tuple(indices_all_trg_cleaned)

        return index_map

    def _get_idx2ngram_trg(
        self,
        ngrams2indices_src: Dict[Tuple[str, ...], List[Tuple[int, ...]]],
        alignment: List[List[int | None]],
        repl_lex: Dict[Tuple[str, ...], Tuple[str, ...]],
    ) -> Dict[Tuple[int, ...], Tuple[str, ...]]:
        """
        Creates a mapping of a source index tuple to a desired source ngram according to
        replacement lexicon

        """
        if not (len(alignment)):
            return {}
        # flatten
        search_src_indices = [
            val for list in ngrams2indices_src.values() for val in list
        ]
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

        Helper function for: _update_target_tok
        """
        start2ngram_and_end = {}
        prev_end = -1
        for idxs, ngram in idx2ngram.items():
            # Not needed for now: remove all None values from idxs
            # idxs = tuple([i for i in idxs if i is not None])
            # if idxs:
            #     continue

            assert len(idxs) > 0
            start = idxs[0]
            end = idxs[-1]
            if remove_overlap and (start <= prev_end):
                # TODO: log; return identifiers (basename, sent_id)
                logger.info(
                    f"Dropped ngram at position [{start} : {end}] because of overlap with previous ngram. Sentence: '{self._current_sample.get(self.raw_trg)}'"
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
    ) -> List[str]:
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

    def map_tokens_cross_layer(
        self,
        tokens_src: List[str],
        tokens_trg: List[str],
        alignment: List[List[int | None]],
    ) -> Tuple[List[str], bool]:
        """
        Returns a modified version of `tokens_trg` in which the cross layer
        type mapping is applied. Returns a tuple `(tokens_new, any_changes)`,
        where `any_changes` is False iff `tokens_new==tokens_old`.
        """
        tokens_trg_new = []
        search_ngrams_src = set(self.replacement_mapping.keys())
        # Find source ngrams in source tokens
        ngram2idxs_src = self._find_ngram_indices(
            search_ngrams_src,
            tokens_src,
            self.src_ngram_lengths,
        )
        # Nothing to change: exit
        if not ngram2idxs_src:
            return tokens_trg, False
        # Find positions in target ngrams where new target ngrams will go
        idx2ngram_trg = self._get_idx2ngram_trg(
            ngram2idxs_src, alignment, self.replacement_mapping
        )
        # Apply changes to target tokens
        tokens_trg_new = self._update_target_tok(
            tokens_trg, idx2ngram_trg, remove_overlap=True
        )
        return tokens_trg_new, True

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        Here the modification is applied to {layer}_tok, and the changes are
        propagated to {layer}_raw, etc.
        """
        self._current_sample = sample

        # Skip samples that have been classified as non-German
        # Possible TODO: allow more and flexible conditions
        # instead of hard-coded
        if self.lang_de_score in sample:
            if sample[self.lang_de_score] == 0:
                return sample

        tokens_trg_old = sample[self.tok_trg]
        tokens_src = sample[self.tok_src]
        alignment = sample[self.alignment]
        tokens_trg_new, any_changes = self.map_tokens_cross_layer(
            tokens_src, tokens_trg_old, alignment
        )
        if any_changes:
            sample[self.tok_trg] = tokens_trg_new
            self.update_raw_from_tok(
                sample,
                key_raw=self.raw_trg,
                key_tok=self.tok_trg,
            )
            self.update_spans_and_ws_from_tok_and_raw(
                sample,
                key_tokens=self.tok_trg,
                key_raw=self.raw_trg,
                key_ws=self.ws_trg,
                key_spans=self.spans_trg,
            )
            self.update_alignment(
                sample,
                key_tokens_src=self.tok_src,
                key_tokens_trg=self.tok_trg,
                key_alignment=self.alignment,
            )

        return sample

    def _load_n2m_replacement_mapping(
        self, files: List[str], delimiters: Optional[str] = None
    ) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
        all_pairs = []
        for file in files:
            with open(file, newline="") as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiters)
                dialect.quotechar = "`"
                csvfile.seek(0)
                reader = csv.reader(csvfile, dialect)
                for row in reader:
                    pair = tuple(row)
                    assert len(pair) == 2, print(pair)
                    src = tuple(pair[0].split(" "))
                    trg = tuple(pair[1].split(" "))
                    all_pairs.append((src, trg))
        replacement_mapping = dict(all_pairs)
        # Set src_ngram_lengths
        self.src_ngram_lengths = self.get_ngram_lengths(replacement_mapping)
        return replacement_mapping
