#!/usr/bin/env python3

"""
Example call:
make_dataset.py --maker dtaevalmaker --data /home/bracke/data/DTAEvalCorpus/orig/xml --metadata /home/bracke/data/dta/metadata/jsonl/metadata_dtak.jsonl --output-dir /home/bracke/data/DTAEvalCorpus/v01
"""

import argparse
from datetime import datetime
import time
from typing import Any, List, Optional

def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-line script for creating a sentence-aligned corpus from its original format. Assumes that there is a maker class for this corpus in which the rules for creation are defined."
    )

    parser.add_argument(
        "-m",
        "--maker",
        help="Name of the maker class",
    )

    parser.add_argument(
        "--data",
        help="Path to the input data directory",
    )

    parser.add_argument(
        "--metadata",
        help="Path to the input metadata file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output directory",
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    # (1) Read arguments
    args = parse_arguments(arguments)

    # (2) Check arguments
    # TODO
    input_dir_data = args.data
    input_dir_metadata = args.metadata
    output_dir = args.output_dir
    plugin = args.maker

    # (3) Select plugin, run maker and save 
    if plugin.lower() == "dtaevalmaker":
        from transnormer_data.maker.dta_eval_maker import DtaEvalMaker
        maker: Any = DtaEvalMaker(
            input_dir_data, input_dir_metadata, output_dir
        )
    elif plugin.lower() == "dtakmaker":
        from transnormer_data.maker.dtak_maker import DtakMaker
        maker = DtakMaker(
            input_dir_data, input_dir_metadata, output_dir
        )
    dataset = maker.make(save=True)

if __name__ == "__main__":

    print(f"Current time: {datetime.now().time()}")
    t = time.process_time()
    main()
    elapsed_time = time.process_time() - t
    print(f"Process took: {elapsed_time}")
    print(f"Current time: {datetime.now().time()}")