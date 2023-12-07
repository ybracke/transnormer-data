import json
import os
from typing import Dict, List, Optional, Union

import datasets

from transnormer_data.base_maker import BaseMaker
from transnormer_data.base_dataset_modifier import BaseDatasetModifier


class DtaMaker(BaseMaker):
    """Common parent class for DtaEvalMaker, DtakMaker and DtaeMaker"""

    def __init__(
        self,
        path_data: Union[str, os.PathLike],
        path_metadata: Union[str, os.PathLike],
        path_output: Union[str, os.PathLike],
    ) -> None:
        """Initialize the maker with paths to the data files, metadata file and output directory"""
        self.path_data = path_data
        self.path_metadata = path_metadata
        self.path_output = path_output

        self._dataset: Optional[datasets.Dataset] = None
        self._metadata: Optional[Dict[str, Dict]] = None

        self._modifier: Optional[BaseDatasetModifier] = None

    def _load_metadata(self) -> Dict[str, Dict]:
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

        return metadata_mapper

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

    @staticmethod
    def join_wrongly_splitted_tokens(tokens: List[str]) -> List[str]:
        """Joins strings in a list that should actually be a single string

        Two tokens are considered to be wrongly splitted into two tokens, iff
        (1) the first token starts with a capital letter and ends with a hyphen character
        (2) the second token starts with a capital letter
        """

        # Check if any of the orig tokens should actually be joined into one
        # Example: ['Stände-', 'Verſammlungen'] -> ['Stände-Versammlungen]

        tokens_edited = []
        i = 0

        while i < len(tokens):
            current_token = tokens[i]
            # unify hyphen
            current_token = current_token.replace("¬", "-")

            # Check if the current token ends with a hyphen, starts with capital letter and if the next token starts with a capital letter
            while (
                current_token
                and current_token[-1] == "-"
                and current_token[0].isupper()
                and i + 1 < len(tokens)
                and tokens[i + 1]
                and tokens[i + 1][0].isupper()
            ):
                # Join the current token and the next token
                current_token = current_token + tokens[i + 1]
                i += 1

            tokens_edited.append(current_token)
            i += 1

        return tokens_edited
