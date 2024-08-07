import argparse
import glob
import logging
import math
import os

from typing import List, Optional

import numpy as np
import datasets

from transnormer_data import utils

SEED = 42

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="subsample-dataset.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)

logger.setLevel(logging.INFO)


def filter_for_charlength(
    ds: datasets.Dataset,
    length_lower_bound: int = 0,
    length_upper_bound: Optional[int] = None,
) -> datasets.Dataset:
    """
    Keep only the lines which lengths lie between upper and lower bound.
    """
    if length_upper_bound is None:
        length_upper_bound = 100_000_000
    ds = ds.filter(
        lambda example: len(example["orig"]) > length_lower_bound
        and len(example["orig"]) <= length_upper_bound
    )
    return ds


def subsample(
    ds: datasets.Dataset, p_samples: float, p_short: float
) -> datasets.Dataset:
    """
    Create a subsample of `dataset` where only `p_samples` of all samples in the dataset are retained.

    The retained samples are selected based on their character length:
        * A fraction of `p_short` of the samples has a length up until the Q1 (lowest 25 %)
        * A fraction of 1.0 - `p_short` of the samples has a length that lies within the interquartile range (IQR, medium 50 %) of the data
    """

    # Process dataset, compute length statistics and do sampling

    # Extract data
    n_sents = len(ds["orig"])
    orig_lengths = [len(s) for s in ds["orig"]]
    # lm_scores = ds["dbmdz/german-gpt2"]
    # lm_scores = [score for score in lm_scores if score is not None]
    # year = ds["date"][0]
    # basename = ds["basename"][0]

    # Compute number of output sentences
    n_sents_subsample = math.floor(n_sents * p_samples)
    n_short = math.floor(n_sents_subsample * p_short)
    n_iqr = n_sents_subsample - n_short

    # Compute statistics
    Q1_sent_length = math.floor(np.percentile(orig_lengths, 25))
    Q3_sent_length = math.floor(np.percentile(orig_lengths, 75))

    # Create two filtered version of ds
    ds_filtered_short = filter_for_charlength(ds, 0, Q1_sent_length)
    ds_filtered_iqr = filter_for_charlength(ds, Q1_sent_length, Q3_sent_length)

    # From the filtered versions sample the final version
    ds_sampled_short = ds_filtered_short.shuffle(seed=SEED).select(range(n_short))
    ds_sampled_iqr = ds_filtered_iqr.shuffle(seed=SEED).select(range(n_iqr))
    ds_final = datasets.concatenate_datasets([ds_sampled_short, ds_sampled_iqr])

    # Sort
    ds_final = ds_final.sort("par_idx")

    return ds_final


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        arguments (Optional[List[str]]): List of command-line arguments (default is None).

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the input data file or directory. If path is a directory, script will handle all *.jsonl files below that path.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to the output directory for subsampled files.",
    )

    parser.add_argument(
        "--proportion-subsample",
        type=float,
        default=0.1,
        help="Proportion of sentences of the total data set that go into the subsample (default: %(default)s).",
    )

    parser.add_argument(
        "--proportion-short-samples",
        type=float,
        default=0.1,
        help="Proportion of short sentences in the subsample (default: %(default)s). The rest of the subsample will be made up by sentences whose character lengths lie in the IQR of the dataset's sentence lengths.",
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    """
    Main function to perform subsampling of datasets.

    Args:
        arguments (Optional[List[str]]): List of command-line arguments (default is None).
    """
    args = parse_arguments(arguments)

    input_path = args.data
    output_dir = args.output_dir
    p_sents = args.proportion_subsample
    p_short = args.proportion_short_samples

    # (2) Get data files
    # Default: Put every file into its own bin -
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

    # (3) Iterate over files lists, modify, save
    for files in files_lists:
        # (3.1) Load dataset
        logger.info("Handling: " + " ".join(files))
        dataset: datasets.Dataset = utils.load_dataset_via_pandas(data_files=files)
        dataset.data.validate()

        # (3.2) Get subsampled dataset
        dataset_subsample = subsample(dataset, p_sents, p_short)

        # (3.3) Save dataset
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        utils.save_dataset_to_json_grouped_by_property(
            dataset_subsample, property="basename", path_outdir=output_dir
        )


if __name__ == "__main__":
    main()
