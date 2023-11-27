import json
import os
import glob
from typing import Dict, List, Tuple, Union

import datasets
from lxml import etree


def get_basename_no_ext(file_path: Union[str, os.PathLike]) -> str:
    try:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return basename
    except IndexError:
        return ""


def join_metadata(
    dataset: datasets.Dataset, metadata: Dict[str, Dict], join_on: str
) -> datasets.Dataset:
    """ """
    # the following assumes all entries have the same structure
    new_columns = {key: [] for key in list(metadata.values())[0] if key != join_on}
    join_column = dataset[join_on]
    for key in join_column:  # DTA Eval: key = basename
        metadata_for_this_key: Dict = metadata.get(key)
        for key, value in metadata_for_this_key.items():
            if key != join_on:
                new_columns[key].append(value)

    for name, column in new_columns.items():
        dataset = dataset.add_column(name, column)

    return dataset


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

        self._data = None
        self._metadata = None
        self._dataset = None

    def make(self) -> datasets.Dataset:
        self._metadata = self.load_metadata()
        self._data = self.load_data()
        self._dataset = join_metadata(self._data, self._metadata, join_on="basename")
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
            basename = get_basename_no_ext(fname_in)
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

    def save(self, docwise: bool = True) -> None:
        """Save dataset, either in a single file, or document-wise as OUTDIR/{basename}.jsonl"""
        if not os.path.isdir(self.path_output):
            os.makedirs(self.path_output)
        if not docwise:
            self._dataset.to_json(
                os.path.join(self.path_output, "dataset.jsonl")
            )  # TODO
        # Save dataset document-wise
        else:
            basename = None  # current basename
            f = None
            for row in self._dataset:
                # open a new file when the basenam changed, start writing to it
                if row["basename"] != basename:
                    if f is not None:
                        f.close()
                    basename = row["basename"]
                    filename = os.path.join(self.path_output, f"{basename}.jsonl")
                    f = open(filename, "w", encoding="utf-8")
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                # otherwise just write line
                else:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
