import json
import os
from typing import Dict, List, Tuple, Union

import datasets


def get_basename_no_ext(file_path: Union[str, os.PathLike]) -> str:
    try:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return basename
    except IndexError:
        return ""


def save_dataset_to_json_grouped_by_property(
    dataset: datasets.Dataset, property: str, path_outdir: Union[str, os.PathLike]
) -> None:
    """Save a datasets.Dataset into multiple JSONL files grouped by a common value of property

    `property` must be a column in the dataset. The common value by which records are grouped will be used as the output filename, i.e. the path will be "path_outputdir/{value_property}.jsonl". For example, if the "basename" property is taken (for DTA EvalCorpus and DTAK), the output path can be "path/to/dir/fontane_stechlin_1899.jsonl"

    `path_outdir` must be an existing directory path
    """
    value_property = None  # current basename
    f = None
    for row in dataset:
        # open a new file when the basenam changed, start writing to it
        if row[property] != value_property:
            if f is not None:
                f.close()
            value_property = row[property]
            filename = os.path.join(path_outdir, f"{value_property}.jsonl")
            f = open(filename, "w", encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # otherwise just write line
        else:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
