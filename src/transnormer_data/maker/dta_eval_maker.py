import json
import os
import glob
import re
from typing import Dict, List, Optional, Tuple, Union

import datasets
from lxml import etree

from transnormer_data import utils
from transnormer_data.maker.dta_maker import DtaMaker
from transnormer_data.modifier.vanilla_dtaeval_modifier import VanillaDtaEvalModifier


class DtaEvalMaker(DtaMaker):
    """An object that creates a dataset in the transnormer format from the DTA Eval Corpus in its original XML format, plus metadata in JSONL format"""

    def __init__(
        self,
        path_data: Union[str, os.PathLike],
        path_metadata: Union[str, os.PathLike],
        path_output: Union[str, os.PathLike],
    ) -> None:
        """Initialize the maker with paths to the data files, metadata file and output directory"""
        super().__init__(path_data,path_metadata,path_output)

        self._modifier: Optional[VanillaDtaEvalModifier] = None

    def make(self, save: bool = False) -> datasets.Dataset:
        """Create a datasets.Dataset object from the paths passed to the constructor.

        Pass `save=True` to save the dataset in JSONL format to the output directory that was passed to the constructor. If the directory does not exists, it will be created.
        """
        self._metadata = self._load_metadata()
        self._dataset = self._load_data()
        self._dataset = self._join_data_and_metadata(join_on="basename")
        self._modifier = VanillaDtaEvalModifier(self._dataset)
        self._dataset = self._modifier.modify_dataset()
        if save:
            if not os.path.isdir(self.path_output):
                os.makedirs(self.path_output)
            utils.save_dataset_to_json_grouped_by_property(
                self._dataset, property="basename", path_outdir=self.path_output
            )
        return self._dataset

    def _load_data(self) -> datasets.Dataset:
        """
        Reads from a DTA EvalCorpus XML file into a dataset

        """
        basenames = []
        sents_orig_tok = []
        sents_norm_tok = []
        sents_is_bad = []
        sents_orig_tokclass = []
        par_idxs = []
        for fname_in in glob.iglob(os.path.join(self.path_data, "*"), recursive=True):
            basename = utils.get_basename_no_ext(fname_in)
            tree = etree.parse(fname_in)
            # sentences
            for i, s in enumerate(tree.iterfind("//s")):
                (
                    sent_orig_tok,
                    sent_norm_tok,
                    sent_orig_tokclass,
                ) = self._create_example_from_s(s)
                basenames.append(basename)
                sents_orig_tok.append(sent_orig_tok)
                sents_norm_tok.append(sent_norm_tok)
                sents_orig_tokclass.append(sent_orig_tokclass)
                par_idxs.append(i)
                is_bad = True if "sbad" in s.attrib else False
                sents_is_bad.append(is_bad)

        length = len(basenames)
        assert (
            length
            == len(par_idxs)
            == len(sents_orig_tok)
            == len(sents_norm_tok)
            == len(sents_is_bad)
        )
        return datasets.Dataset.from_dict(
            {
                "basename": basenames,
                "par_idx": par_idxs,
                "orig_tok": sents_orig_tok,
                "orig_ws": [None for i in range(length)],
                "orig_class": sents_orig_tokclass,
                "norm_tok": sents_norm_tok,
                "norm_ws": [None for i in range(length)],
                "is_bad": sents_is_bad,
            }
        )

    def _join_data_and_metadata(self, join_on: str) -> datasets.Dataset:
        """Join the metadata (stored in dictionary) with the data (stored in dataset) on a key ('join_on') that is contained in both"""
        assert self._metadata is not None and self._dataset is not None
        # new_columns
        # the following assumes all metadat entries have the same structure
        new_columns: Dict[str, List] = {
            key: [] for key in list(self._metadata.values())[0] if key != join_on
        }
        # Get the column to join metadata and data on, e.g. "basename"
        join_column = self._dataset[join_on]

        for entry in join_column:
            # metadata dictionary for a specific property value
            # e.g. for basename=='fontane_stechlin_1899'
            try:
                metadata = self._metadata[entry]
            except KeyError as e:
                print(e, f"{entry} not in metadata dictionary - check metadata file")
                raise e
            for key, value in metadata.items():
                if key != join_on:
                    new_columns[key].append(value)

        # Append new columns to dataset
        for name, column in new_columns.items():
            self._dataset = self._dataset.add_column(name, column)

        return self._dataset

    def _create_example_from_s(
        self, s: etree.Element
    ) -> Tuple[List[str], List[str], List[str]]:
        """Create an example from a sentence element in an DTAEvalCorpus XML file

        An example is a 3-Tuple:
        sent_orig_tok: List[str]
        sent_norm_tok: List[str]
        sent_orig_tokclass: List[str]

        """

        sent_orig_tok = []
        sent_norm_tok = []
        sent_orig_tokclass = []
        # Iterate over first <w> under <s>
        for w in s.xpath("./w"):
            # 1. Filter tokens that have no @old version (= rare errors)
            try:
                orig = w.attrib["old"]
            except KeyError:
                continue
            # 2. Store @old as normalization, if there is no @new
            try:
                norm = w.attrib["new"]
            except KeyError:
                norm = w.attrib["old"]

            tokclass = w.attrib["class"]  # e.g. LEX, JOIN, BUG

            # 3. Handle "JOIN": Multiple orig tokens (and their annotations) which have a single norm form
            # Get inner w-nodes for orig and outer w-node for norm
            if w.attrib["class"] == "JOIN":
                # get the orig tokens and annos
                tokens = [inner_w.attrib["old"] for inner_w in list(w.iterfind("w"))]
                tokclass_annos = [
                    inner_w.attrib["class"] for inner_w in list(w.iterfind("w"))
                ]
                tokens_mod = self.join_wrongly_splitted_tokens(tokens)

                # If tokens changed, adapt the annotations: use default for remaining token(s)
                if tokens != tokens_mod:
                    tokclass_annos = ["LEX" for t in tokens_mod]

                sent_orig_tok.extend(tokens_mod)
                sent_orig_tokclass.extend(tokclass_annos)
                sent_norm_tok.append(norm)
                continue

            # 4. Handle hyphens
            # Unify hyphen-character
            orig = orig.replace("¬", "-")
            norm = norm.replace("¬", "-")
            # Add missing hyphen on norm
            if orig[-1] == "-" and norm[-1] != "-":
                norm += "-"

            # 5. Split tokens on " " or "_"
            # orig split
            orig_split, orig_was_split = self.custom_split(orig)
            if orig_was_split:
                sent_orig_tokclass.extend(
                    [w.attrib["class"] for i in range(len(orig_split))]
                )
            else:
                sent_orig_tokclass.append(tokclass)
            sent_orig_tok.extend(orig_split)
            # norm split
            norm_split, norm_was_split = self.custom_split(norm)
            sent_norm_tok.extend(norm_split)

        return sent_orig_tok, sent_norm_tok, sent_orig_tokclass

    @staticmethod
    def custom_split(input_string: str) -> Tuple[List[str], bool]:
        """Returns the list of token(s) and True if there actually was a split"""
        if "_" in input_string or " " in input_string:
            return re.split(r"[_\s]+", input_string), True
        else:
            return [input_string], False

