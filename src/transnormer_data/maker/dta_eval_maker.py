import json
import os
import glob
from typing import Dict, List, Optional, Tuple, Union

import datasets
from lxml import etree

from transnormer_data import utils


class DtaEvalMaker:
    def __init__(
        self,
        path_data: Union[str, os.PathLike],
        path_metadata: Union[str, os.PathLike],
        path_output: Union[str, os.PathLike],
    ) -> None:
        self.path_data = path_data
        self.path_metadata = path_metadata
        self.path_output = path_output

        self._dataset: Optional[datasets.Dataset] = None
        self._metadata: Optional[Dict[str, Dict]] = None

    def make(self, save: bool = False) -> datasets.Dataset:
        self._metadata = self.load_metadata()
        self._dataset = self.load_data()
        self._dataset = self.join_data_and_metadata(join_on="basename")
        if not os.path.isdir(self.path_output):
            os.makedirs(self.path_output)
            utils.save_dataset_to_json_grouped_by_property(
                self._dataset, property="basename", path_outdir=self.path_output
            )
        return self._dataset

    def load_metadata(self) -> Dict[str, Dict]:
        """
        Create a metadata_mapper (example below) from JSONL file.

        Example entry JSON:

        {
            "date_": "1867",
            "author": "Auerbach, Berthold (#11865103X)",
            "basename": "auerbach_sanders_1867",
            "bibl": "Auerbach, Berthold: Brief an Daniel Sanders. Bonn, 10. März 1867.",
            "collection": "dtae",
            "firstDate": "1867",
            "textClass": "Gebrauchsliteratur::Brief",
            "title": "Brief an Daniel Sanders"
        }

        Example entry metadata_mapper:

        {
            "auerbach_sanders_1867": {
                # changed "date_" / "firstDate" to "date" and cast to int
                "date": 1867,
                "author": "Auerbach, Berthold (#11865103X)",
                "basename": "auerbach_sanders_1867",
                # popped "bibl" and "collection"
                # changed "textClass_" to "genre"
                "genre": "Gebrauchsliteratur::Brief",
                "title": "Brief an Daniel Sanders"
            }
        }

        """
        metadata_mapper = {}

        with open(self.path_metadata, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                # Assert that the two dates are identical, keep only one, change the variable name
                assert record["date_"] == record["firstDate"]
                record["date"] = int(record["firstDate"])
                record.pop("date_")
                record.pop("firstDate")
                record.pop("bibl")
                record.pop("collection")
                record["genre"] = record.pop("textClass")

                # Make entry to metadata_mapper
                id = record["basename"]
                if id not in metadata_mapper:
                    metadata_mapper[id] = record
                # else:
                #     logging.info(f"Tried to add {id} to metadata dict multiple times.")

        return metadata_mapper

    def load_data(self) -> datasets.Dataset:
        """
        Reads from a DTA EvalCorpus XML file into a dataset

        """
        basenames = []
        sents_orig_tok = []
        sents_norm_tok = []
        sents_is_bad = []
        par_idxs = []
        for fname_in in glob.iglob(os.path.join(self.path_data, "*"), recursive=True):
            basename = utils.get_basename_no_ext(fname_in)
            tree = etree.parse(fname_in)
            # sentences
            for i, s in enumerate(tree.iterfind("//s")):
                sent_orig_tok = []
                sent_norm_tok = []
                # Iterate over first <w> under <s>
                for w in s.xpath("./w"):
                    # Filter tokens that have no @old version (= rare errors)
                    try:
                        orig = w.attrib["old"]
                    except KeyError:
                        continue
                    # Store @old as normalization, if there is none (e.g. "Mur¬" -> "Mur¬")
                    try:
                        norm = w.attrib["new"]
                    except KeyError:
                        norm = w.attrib["old"]

                    sent_orig_tok.append(orig)
                    sent_norm_tok.append(norm)

                basenames.append(basename)
                sents_orig_tok.append(sent_orig_tok)
                sents_norm_tok.append(sent_norm_tok)
                par_idxs.append(i)
                is_bad = True if "sbad" in s.attrib else False
                sents_is_bad.append(is_bad)

        return datasets.Dataset.from_dict(
            {
                "basename": basenames,
                "par_idx": par_idxs,
                "orig_tok": sents_orig_tok,
                "norm_tok": sents_norm_tok,
                "is_bad": sents_is_bad,
            }
        )

    def join_data_and_metadata(self, join_on: str) -> datasets.Dataset:
        """Join the metadata (stored in dictionary) with the data (stored in dataset) on a key ('join_on') that is contained in both"""
        # the following assumes all entries have the same structure
        new_columns = {
            key: [] for key in list(self._metadata.values())[0] if key != join_on
        }
        # Get the column to join metadata and data on, e.g. "basename"
        join_column = self._dataset[join_on]

        for entry in join_column:
            # metadata dictionary for a specific property value
            # e.g. for basename=='fontane_stechlin_1899'
            metadata = self._metadata.get(entry)
            for key, value in metadata.items():
                if key != join_on:
                    new_columns[key].append(value)

        # Append new columns to dataset
        for name, column in new_columns.items():
            self._dataset = self._dataset.add_column(name, column)

        return self._dataset
