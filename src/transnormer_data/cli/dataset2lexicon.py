import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

from tqdm import tqdm

from transnormer_data.utils import german_transliterate, filename_gen

# Reset existing logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="dataset2lexicon.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)

logger.setLevel(logging.INFO)


def transform_alignment(input_alignment):
    """
    Transform the input alignment from (seq1_index, seq2_index) pairs into a grouped output alignment,
    ensuring there are no duplicate tuples.

    Parameters:
    input_alignment (list of tuples): A list where each tuple consists of two integers,
                                       the first being an index from seq1 and the second being an index from seq2.

    Returns:
    list of tuples: The transformed alignment, where each tuple contains two elements:
                     - The first element is a tuple of all seq1 indices that align with the current seq2 index.
                     - The second element is a tuple of all seq2 indices that align with the current seq1 index.
    """
    # Step 1: Create a mapping of seq2 indices to the corresponding seq1 indices
    seq2_to_seq1 = defaultdict(list)
    for seq1_idx, seq2_idx in input_alignment:
        seq2_to_seq1[seq2_idx].append(seq1_idx)

    # Step 2: Create a mapping of seq1 indices to the corresponding seq2 indices
    seq1_to_seq2 = defaultdict(list)
    for seq1_idx, seq2_idx in input_alignment:
        seq1_to_seq2[seq1_idx].append(seq2_idx)

    # Step 3: Build the output list by grouping the indices
    output_alignment = []
    seen_pairs = set()  # To keep track of pairs we've already added to the output

    # Combine both mappings to get unique groups
    unique_seq2_indices = set(seq2_to_seq1.keys())
    for seq2_idx in unique_seq2_indices:
        # Gather all seq1 indices that correspond to seq2_idx
        seq1_indices = tuple(sorted(seq2_to_seq1[seq2_idx]))
        # Gather all seq2 indices that correspond to seq1_idx
        seq2_indices = tuple(
            sorted(seq1_to_seq2[seq1_indices[0]])
        )  # seq1_indices[0] is used to retrieve seq2 indices

        # If this pair of (seq1_indices, seq2_indices) has not been added before, add it to the output
        pair = (seq1_indices, seq2_indices)
        if pair not in seen_pairs:
            output_alignment.append(pair)
            seen_pairs.add(pair)

    # Step 4: Return the final output alignment
    return output_alignment


def get_ngram_alignment(alignment, seq1, seq2):
    ngram_mappings = []
    for indexes_seq1, indexes_seq2 in alignment:
        # TODO: just skip None alignments?
        ngram_seq1 = [seq1[i] for i in indexes_seq1 if i is not None]
        ngram_seq2 = [seq2[i] for i in indexes_seq2 if i is not None]
        ngram_mapping = ("_".join(ngram_seq1), "_".join(ngram_seq2))
        ngram_mappings.append(ngram_mapping)
    return ngram_mappings


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates a lexicon from a dataset with the following fields: \
        `ngram_orig`, `ngram_norm`, `freq`, `docs`"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input data file or directory.",
    )

    parser.add_argument(
        "-x",
        "--transliterate",
        action="store_true",
        help="Whether the 'orig_tok' layer should be transliterated before counting (this would merge 'ſchoͤn' and 'schön'; default=%(default)s).",
    )
    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    # (1) Read and check arguments
    args = parse_arguments(arguments)
    translit = args.transliterate
    files = list(filename_gen(args.data))

    # In how many documents does the ngram mapping occur
    cnt_occurs_in_docs: Counter[Tuple[str, str]] = Counter()
    # How often does the ngram mapping occur overall
    cnt_freqs_all: Counter[Tuple[str, str]] = Counter()

    # Iterate over files
    for path in tqdm(files):
        # Load dataset
        logger.info(f"Handling: {path}")

        cnt_freqs_doc: Counter[Tuple[str, str]] = Counter()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                # assert structure
                # TODO
                alignment = transform_alignment(record["alignment"])
                orig_tok = record["orig_tok"]
                if translit:
                    orig_tok = [german_transliterate(tok) for tok in orig_tok]
                norm_tok = record["norm_tok"]
                ngram_alignment = get_ngram_alignment(alignment, orig_tok, norm_tok)
                cnt_freqs_doc.update(ngram_alignment)

        cnt_occurs_in_docs.update(cnt_freqs_doc.keys())
        cnt_freqs_all.update(cnt_freqs_doc)

        assert len(cnt_freqs_all) == len(cnt_occurs_in_docs)

    # TODO
    # Create an output file in this format:
    # {'orig': str, 'norm': str, 'freq': int, 'docs': int}

    # print(cnt_occurs_in_docs.most_common()[:100])


if __name__ == "__main__":
    """
    Example call currently (output file missing):

    python3 src/transnormer_data/cli/dataset2lexicon.py --data ~/published_datasets/dtak-transnormer-full-v1/data/test/1600-1699/ -x > file.txt
    """
    main()
