import argparse
import glob
import os

from typing import List, Optional

import datasets

from transnormer_data import utils
from transnormer_data.modifier import replace_token_1to1_modifier


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-line script for modifying a dataset stored in one or more JSONL files. Assumes that there is a modifier class for the task."
    )

    parser.add_argument(
        "-m",
        "--modifier",
        help="Name of the modifier class",
    )

    # TODO
    parser.add_argument(
        "--modifier-kwargs",
        help="Arguments as key=value pairs that are passed to the modifier (e.g. replacement files)",
    )

    parser.add_argument(
        "--data",
        help="Path to the input data directory",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output directory",
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:

    # (1) Read and check arguments
    args = parse_arguments(arguments)
    # TODO
    input_dir_data = args.data
    output_dir = args.output_dir
    plugin = args.modifier

    # # Parse --modifier-kwargs into a dictionary
    # if args.modifier_kwargs:
    #     modifier_kwargs = dict(item.split("=") for item in args.modifier_kwargs.split())

    # (2) Get data files
    # TODO: Perhaps exchange the part in DtakMaker.make with this code
    # len(files_list) == number_of_files
    # Below, this will overwrite `dataset` with every iteration
    files_list: List[List[str]] = sorted(
        [[fname] for fname in glob.iglob(os.path.join(input_dir_data, "*"), recursive=True)]
    )
    # Prevent this, if desired --> len(files_list) == 1
    if args.merge_into_single_dataset:
        files_list = [[fname for fname in files_list[0]]]

    # (3) Create modifier
    if plugin == "replacetoken1to1modifier":
        modifier = replace_token_1to1_modifier.ReplaceToken1to1Modifier(
            mapping_files=[] # TODO
            )

    # (4) Iterate over files lists, modify, save
    for files in files_list:
        dataset = datasets.load_dataset("json", data_files=files, split="train")
        dataset = modifier.modify_dataset(dataset)

        # (4.x) Save
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        utils.save_dataset_to_json_grouped_by_property(
            dataset, property="basename", path_outdir=output_dir
            )

    return None


if __name__ == "__main__":
    main()
