import argparse
import glob
import os
import time

from datetime import datetime
from typing import List, Optional

import datasets

from transnormer_data import utils
from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.modifier import (
    replace_token_1to1_modifier,
    replace_token_1ton_modifier,
    replace_ntom_cross_layer_modifier,
    language_tool_modifier,
)


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

    parser.add_argument(
        "--merge-into-single-dataset",
        action="store_true",
        help="Set `merge_into_single_dataset` to True (default: False) when you have a small dataset. Per default we expect the DTAK dataset to be too large to put all incoming documents into a single dataset that is then processed as one. Instead we produce create and save individual dataset objects and run the processing separately on each one of them. Whether `merge_into_single_dataset` is True or False does not make a difference to the saved output files. This is also why, in the future, we might remove the option to merge all incoming files into a single dataset.",
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    # (1) Read and check arguments
    args = parse_arguments(arguments)
    input_dir_data = args.data
    output_dir = args.output_dir
    plugin = args.modifier
    # Parse --modifier-kwargs into a dictionary
    if args.modifier_kwargs:
        modifier_kwargs = dict(item.split("=") for item in args.modifier_kwargs.split())

    # (2) Get data files
    # Default: Put every file into its own bin -> modifier will look at and store
    # each file individually
    # Set args.merge_into_single_dataset to True, if you want all files to be processed # as a single dataset
    files_list: List[List[str]] = sorted(
        [
            [fname]
            for fname in glob.iglob(os.path.join(input_dir_data, "*"), recursive=True)
        ]
    )  # len(files_list) == number_of_files
    if args.merge_into_single_dataset:
        files_list = [[fname for fname in files_list[0]]]  # len(files_list) == 1

    # (3) Create modifier
    if plugin.lower() == "replacetoken1to1modifier":
        mapping_files = modifier_kwargs["mapping_files"].split(",")
        layer = modifier_kwargs["layer"]
        modifier: BaseDatasetModifier = (
            replace_token_1to1_modifier.ReplaceToken1to1Modifier(
                layer=layer, mapping_files=mapping_files
            )
        )

    elif plugin.lower() == "replacetoken1tonmodifier":
        mapping_files = modifier_kwargs["mapping_files"].split(",")
        layer = modifier_kwargs["layer"]
        modifier = replace_token_1ton_modifier.ReplaceToken1toNModifier(
            layer=layer, mapping_files=mapping_files
        )

    elif plugin.lower() == "replacentomcrosslayermodifier":
        mapping_files = modifier_kwargs["mapping_files"].split(",")
        delim = modifier_kwargs["delimiter"]
        source_layer = modifier_kwargs["source_layer"]
        target_layer = modifier_kwargs["target_layer"]
        # If the delimiter is the tab character we need a little hack
        # We have to pass the string "{TAB}" on the command line
        if delim == "{TAB}":
            delim = "\t"
        modifier = replace_ntom_cross_layer_modifier.ReplaceNtoMCrossLayerModifier(
            source_layer=source_layer,
            target_layer=target_layer,
            mapping_files=mapping_files,
            mapping_files_delimiters=delim,
        )

    elif plugin.lower() == "languagetoolmodifier":
        rule_file = modifier_kwargs["rule_file"]
        modifier = language_tool_modifier.LanguageToolModifier(rule_file=rule_file)

    # (4) Iterate over files lists, modify, save
    for files in files_list:
        # (4.1) Load dataset
        dataset: datasets.Dataset = utils.load_dataset_via_pandas(data_files=files)
        dataset.data.validate()

        # (4.2) Modify dataset
        dataset = modifier.modify_dataset(dataset)

        # (4.3) Save dataset
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        utils.save_dataset_to_json_grouped_by_property(
            dataset, property="basename", path_outdir=output_dir
        )

    return None


if __name__ == "__main__":
    print(f"Current time: {datetime.now().time()}")
    t = time.process_time()
    main()
    elapsed_time = time.process_time() - t
    print(f"Process took: {elapsed_time}")
    print(f"Current time: {datetime.now().time()}")
