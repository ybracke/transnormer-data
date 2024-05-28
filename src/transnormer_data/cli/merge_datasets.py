import argparse
import glob
import json
import os
import time
from datetime import datetime
from typing import List, Optional

import datasets
import pandas as pd

from transnormer_data import utils


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    """ Command-line script for merging two datasets with matching records.
    
    Usage:
    python3 src/transnormer-data/cli/merge_datasets.py  \
        --on [$COLUMN ...] \
        --ds1 $DATASET1 \
        --ds2 $DATASET2 \
        --out $OUTPATH 
    """

    parser = argparse.ArgumentParser(description="...")

    parser.add_argument(
        "--on",
        help="Name of columns that serve as unique identifier for records, i.e. on which records are matched and merged.",
        nargs="+",
    )

    parser.add_argument("--ds1", help="Path to dataset1 (JSONL file)")

    parser.add_argument("--ds2", help="Path to dataset2 (JSONL file)")

    parser.add_argument("--out", help="Path to output (JSONL file)")

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_arguments((arguments))

    uid: List[str] = args.on  # e.g. ('basename','par_idx')

    dataset1 = pd.read_json(args.ds1, lines=True)
    dataset2 = pd.read_json(args.ds2, lines=True)

    joined_dataset = pd.merge(dataset1, dataset2, on=uid)

    utils.save_pandas_to_jsonl(joined_dataset, args.out)

    return


if __name__ == "__main__":
    main()
