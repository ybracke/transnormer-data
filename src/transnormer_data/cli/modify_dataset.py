import argparse
from typing import List, Optional

import datasets


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument(
        "-x",
        "--long-argument",
        help="...",
    )

    parser.add_argument("dir_in", help="Path to the input directory")

    parser.add_argument("dir_out", help="Path to the output directory")

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    # (1) Read arguments
    args = parse_arguments(arguments)

    # (2) Load dataset
    ds = datasets.load_dataset("json", data_files=args.dir_in)

    # (3) Modify dataset
    ds.map(modify_record, fn_kwargs={"functions": args.plugins}, batched=False)

    # (4) Write the modified version of the dataset

    return None


if __name__ == "__main__":
    main()
