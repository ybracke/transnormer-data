import argparse
import glob
import logging
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
    replace_raw_modifier,
    language_tool_modifier,
    language_detection_modifier,
    lm_score_modifier,
    case_modifier,
)

# Reset existing logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="modify-dataset.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)

logger.setLevel(logging.INFO)


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
        type=str,
        help="Path to the input data file or directory.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output directory. Directory will be created, if it does not exist.",
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
    input_path = args.data
    output_dir = args.output_dir
    plugin = args.modifier
    # Parse --modifier-kwargs into a dictionary
    if args.modifier_kwargs:
        modifier_kwargs = dict(item.split("=") for item in args.modifier_kwargs.split())
    else:
        modifier_kwargs = {}

    # (2) Get data files
    # Default: Put every file into its own bin -> modifier will look at and store
    # each file individually
    # Set args.merge_into_single_dataset to True, if you want all files to be processed # as a single dataset
    # Accepts directory (where it looks for all *.jsonl files) or file path
    if os.path.isdir(input_path):
        files_lists: List[List[str]] = sorted(
            [
                [fname]
                for fname in sorted(
                    glob.iglob(os.path.join(input_path, "**"), recursive=True)
                )
                if fname.endswith(".jsonl")
            ]
        )
    elif os.path.isfile(input_path):
        files_lists = [[input_path]]
    else:
        raise ValueError(f"Unknown path: '{input_path}'")

    if args.merge_into_single_dataset:
        files_lists = [[fname for fname in files_lists[0]]]

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

    elif plugin.lower() == "replacerawmodifier":
        mapping_files = modifier_kwargs["mapping_files"].split(",")
        layer = modifier_kwargs["layer"]
        # uid_labels = modifier_kwargs["uid_labels"]
        raw_label = modifier_kwargs["raw_label"]
        modifier = replace_raw_modifier.ReplaceRawModifier(
            layer=layer,
            mapping_files=mapping_files,
            # uid_labels=uid_labels,
            raw_label=raw_label,
        )

    elif plugin.lower() == "languagetoolmodifier":
        rule_file = modifier_kwargs["rule_file"]
        modifier = language_tool_modifier.LanguageToolModifier(rule_file=rule_file)

    elif plugin.lower() == "languagedetectionmodifier":
        layer = modifier_kwargs.get("layer")
        modifier = language_detection_modifier.LanguageDetectionModifier(layer)

    elif plugin.lower() == "lmscoremodifier":
        layer = modifier_kwargs.get("layer")
        model = modifier_kwargs.get("model")
        modifier = lm_score_modifier.LMScoreModifier(layer, model)

    elif plugin.lower() == "casemodifier":
        layer = modifier_kwargs.get("layer")
        model = modifier_kwargs.get("model")
        batch_size = 8  # TODO: replace hard-coded batch size
        modifier = case_modifier.CaseModifier(layer, model)

    else:
        raise ValueError(
            f"Unknown modifier name '{plugin}'. Please select a valid modifier name."
        )

    # (4) Iterate over files lists, modify, save
    for files in files_lists:
        # (4.1) Load dataset
        logger.info("Handling: " + " ".join(files))
        dataset: datasets.Dataset = utils.load_dataset_via_pandas(
            data_files=files
        )  # type:ignore
        dataset.data.validate()

        # (4.2) Modify dataset
        if "batch_size" in locals():
            # Sort by length for faster generation
            index_column = "#"
            dataset = utils.sort_dataset_by_length(
                dataset,
                column="orig",  # TODO
                descending=False,
                name_index_column=index_column,
                keep_length_column=False,
            )
            dataset = modifier.modify_dataset(dataset, batch_size=batch_size)
            # Restore original order
            dataset = dataset.sort(index_column)
            dataset = dataset.remove_columns(index_column)
        else:
            dataset = modifier.modify_dataset(dataset)

        # (4.3) Save dataset
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        utils.save_dataset_to_json_grouped_by_property(
            dataset, property="basename", path_outdir=output_dir
        )

    return None


if __name__ == "__main__":
    logger.info(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    t = time.process_time()
    main()
    elapsed_time = time.process_time() - t
    logger.info(f"Process took: {elapsed_time:.2f} seconds.")
    logger.info(f"End time: {datetime.now().strftime('%H:%M:%S')}")
