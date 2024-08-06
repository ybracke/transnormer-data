import argparse
import glob
import logging
import os

from typing import List, Optional

import numpy as np
import datasets

from transnormer_data import utils

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="subsample-dataset.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)

logger.setLevel(logging.INFO)


def subsample(
    dataset: datasets.Dataset, p_samples: float, p_short: float
) -> datasets.Dataset:
    """
    Create a subsample of `dataset` where only `p_samples` of all samples in the dataset are retained.

    The retained samples are selected based on their character length:
        * A fraction of `p_short` of the samples has a length up until the Q1 (lowest 25 %)
        * A fraction of 1.0 - `p_short` of the samples has a length that lies within the interquartile range (IQR, medium 50 %) of the data
    """

    # Process dataset, compute length statistics and do sampling

    # Extract data
    lm_scores = dataset["dbmdz/german-gpt2"]
    # lm_scores = [score for score in lm_scores if score is not None]
    n_sents = len(dataset["orig"])
    orig_lengths = [len(s) for s in dataset["orig"]]
    year = dataset["date"][0]
    basename = dataset["basename"][0]

    # Compute statistics
    Q1_sent_length = np.percentile(orig_lengths, 25)
    median_sent_length = np.percentile(orig_lengths, 50)
    Q3_sent_length = np.percentile(orig_lengths, 75)

    # TODO
    pass


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

    # (4) Iterate over files lists, modify, save
    for files in files_lists:
        # (4.1) Load dataset
        logger.info("Handling: " + " ".join(files))
        dataset: datasets.Dataset = utils.load_dataset_via_pandas(data_files=files)
        dataset.data.validate()

        # (4.2) Get subsampled dataset
        dataset_subsample = subsample(dataset, p_sents, p_short)

        # (4.3) Save dataset
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        utils.save_dataset_to_json_grouped_by_property(
            dataset_subsample, property="basename", path_outdir=output_dir
        )


if __name__ == "__main__":
    main()
