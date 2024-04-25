import os
import json
import re
import shutil
import argparse
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, List, Optional, Tuple


def load_document_metadata(
    folder: str, condition: Callable[[Dict], bool]
) -> List[Dict]:
    """
    Load documents' metadata from JSONL files in the specified folder.

    Args:
        folder (str): Path to the folder containing JSONL files.
        condition (Callable[[Dict], bool]): A function that takes a dictionary representing document metadata
            and returns True if the document meets the condition for inclusion, False otherwise.

    Returns:
        List[Dict]: List of dictionaries where each dictionary represents the metadata of the documents.
    """
    documents_meta = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r") as f:
                first_line = json.loads(f.readline())
                if condition(first_line):
                    document_meta = {
                        "filepath": filepath,
                        "basename": first_line["basename"],
                        "date": first_line["date"],
                        "genre": re.sub(":.+$", "", first_line["genre"].split("::")[0]),
                        "author": first_line["author"],
                    }
                    documents_meta.append(document_meta)
    return documents_meta


def get_decade(year: int) -> int:
    """
    Get the decade from the given year.

    Args:
        year (int): Year of publication.

    Returns:
        int: Decade of publication.
    """
    return (year // 10) * 10


def group_documents_by_decade_genre(
    documents: List[Dict],
) -> Dict[Tuple[int, str], List[Dict]]:
    """
    Group documents by decade of publication and genre.

    Args:
        documents (List[Dict]): List of dictionaries representing documents.

    Returns:
        Dict[Tuple[int, str], List[Dict]]: Dictionary where keys are tuples of (decade, genre)
        and values are lists of documents.
    """
    groups = defaultdict(list)
    for doc in documents:
        group_key = (get_decade(doc["date"]), doc["genre"])
        groups[group_key].append(doc)
    return groups


def split_authors(
    authors: List[str], num_works_per_author: Counter, size_test: float
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split authors into train, validation, and test sets.

    Args:
        authors (List[str]): List of authors.
        num_works_per_author (Counter): Counter object containing number of works per author.
        test_size (float): Size of the dev and test set (default is 0.1; i.e. 10% each).

    Returns:
        Tuple[List[str], List[str], List[str]]: Three lists representing train, validation, and test authors.
    """
    authors_sorted_by_works = sorted(authors, key=lambda x: num_works_per_author[x])
    train_authors, test_authors = train_test_split(
        authors_sorted_by_works, test_size=size_test * 2, random_state=42
    )
    val_authors, test_authors = train_test_split(
        test_authors, test_size=0.5, random_state=42
    )
    return train_authors, val_authors, test_authors


def create_splits(
    groups: Dict[Tuple[int, str], List[Dict]], size_test: float
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train, validation, and test splits from grouped documents.

    Args:
        groups (Dict[Tuple[int, str], List[Dict]]): Dictionary where keys are tuples of (decade, genre) and values are lists of documents.

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: Three lists representing train, validation, and test documents.
    """
    train_docs, val_docs, test_docs = [], [], []
    all_authors: Counter = Counter()
    for group_key, docs in groups.items():
        authors = sorted(set(doc["author"] for doc in docs))
        all_authors.update(authors)
    train_authors, val_authors, test_authors = split_authors(
        list(all_authors.keys()), all_authors, size_test
    )
    for group_key, docs in groups.items():
        for doc in docs:
            if doc["author"] in train_authors:
                train_docs.append(doc)
            elif doc["author"] in val_authors:
                val_docs.append(doc)
            else:
                test_docs.append(doc)

    return train_docs, val_docs, test_docs


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
        "--input-dir",
        required=True,
        help="Path to the input directory containing JSONL files.",
    )
    parser.add_argument(
        "--output-dir-train",
        required=True,
        help="Path to the output directory for train set files.",
    )
    parser.add_argument(
        "--output-dir-validation",
        required=True,
        help="Path to the output directory for validation set files.",
    )
    parser.add_argument(
        "--output-dir-test",
        required=True,
        help="Path to the output directory for test set files.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        help="Earliest publication year for data in the split.",
    )

    parser.add_argument(
        "--year-end", type=int, help="Latest publication year for data in the split."
    )

    parser.add_argument(
        "--size-test",
        type=float,
        default=0.1,
        help="Size of the validation set and test set (default is 0.1).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually move any files, but just print the split",
    )
    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    """
    Main function to perform data splitting.

    Args:
        arguments (Optional[List[str]]): List of command-line arguments (default is None).
    """
    args = parse_arguments(arguments)

    condition = lambda x: args.year_start <= x["date"] <= args.year_end
    documents_meta = load_document_metadata(args.input_dir, condition)
    groups = group_documents_by_decade_genre(documents_meta)
    train_docs, val_docs, test_docs = create_splits(groups, args.size_test)

    if args.dry_run:
        print("# train")
        for doc in sorted(train_docs, key=lambda x: x["basename"]):
            print(doc["basename"], get_decade(doc["date"]), doc["genre"])
        print("\n# validation")
        for doc in sorted(val_docs, key=lambda x: x["basename"]):
            print(doc["basename"], get_decade(doc["date"]), doc["genre"])
        print("\n# test")
        for doc in sorted(test_docs, key=lambda x: x["basename"]):
            print(doc["basename"], get_decade(doc["date"]), doc["genre"])

    else:
        # Create output directories
        os.makedirs(args.output_dir_train, exist_ok=True)
        os.makedirs(args.output_dir_validation, exist_ok=True)
        os.makedirs(args.output_dir_test, exist_ok=True)

        # Move documents
        for docs, dir in [
            (train_docs, args.output_dir_train),
            (val_docs, args.output_dir_validation),
            (test_docs, args.output_dir_test),
        ]:
            for doc_meta in docs:
                filepath = doc_meta["filepath"]
                basename = os.path.basename(filepath)
                dst = os.path.join(dir, basename)
                shutil.move(filepath, dst)


if __name__ == "__main__":
    main()
