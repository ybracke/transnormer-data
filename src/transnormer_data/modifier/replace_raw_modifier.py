import csv
from typing import Dict, List, Optional, Tuple

import spacy

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data.detokenizer import DtaEvalDetokenizer


def infer_type(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


class ReplaceRawModifier(BaseDatasetModifier):
    def __init__(
        self,
        layer: str = "norm",
        mapping_files: Optional[List[str]] = None,
        uid_labels: List[str] = ["basename", "par_idx"],
        raw_label: str = "norm",
    ) -> None:
        """
        This modifier replaces the raw text version on the target layer with a
        corrected string and propagates the changes to the tokenized version.
        """

        # Keys in the sample dictionary
        valid_layers = {"norm", "orig"}
        if layer not in valid_layers:
            raise ValueError(
                f"ReplaceToken1to1Modifier: layer must be one of{valid_layers}"
            )
        self.raw = f"{layer}"
        self.tok = f"{layer}_tok"
        self.ws = f"{layer}_ws"
        self.spans = f"{layer}_spans"
        other_layer = (valid_layers - {layer}).pop()
        self.tok_src = f"{other_layer}_tok"
        self.alignment = "alignment"

        # Detokenizer
        self.detokenizer = DtaEvalDetokenizer()

        # NLP
        self.nlp = spacy.blank("de")

        # Labels of the unique identifier for the following mapping
        self.uid_labels: List[str] = uid_labels

        # Corrected raw samples
        mapping_files = [] if mapping_files is None else mapping_files
        self.corrected_raw_samples: Dict[Tuple[str | int, ...], str] = (
            self._load_corrected_samples(mapping_files, self.uid_labels, raw_label)
        )

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        E.g., if the modification was applied to norm_raw,
        the changes have to be propagated to norm_tok, etc.
        """
        uid = tuple([sample[uid_label] for uid_label in self.uid_labels])
        if uid not in self.corrected_raw_samples:
            return sample
        sample[self.raw] = self.corrected_raw_samples[uid]
        self.update_tok_from_raw(
            sample, key_raw=self.raw, key_tok=self.tok, key_ws=self.ws
        )
        self.update_spans_and_ws_from_tok_and_raw(
            sample,
            key_tokens=self.tok,
            key_raw=self.raw,
            key_ws=self.ws,
            key_spans=self.spans,
        )
        self.update_alignment(
            sample,
            key_tokens_src=self.tok_src,
            key_tokens_trg=self.tok,
            key_alignment=self.alignment,
        )

        return sample

    def _load_corrected_samples(
        self, files: List[str], uid_labels: List[str], raw_label: str
    ) -> Dict[Tuple[str | int, ...], str]:
        mapping = {}
        for fname in files:
            if fname.endswith(".csv") or fname.endswith(".tsv"):
                with open(fname, newline="") as csvfile:
                    dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters="\t,")
                    # dialect.quotechar = "`"
                    csvfile.seek(0)
                    reader = csv.reader(csvfile, dialect)
                    column_names = next(reader)
                    if not (
                        all([(uid_label in column_names) for uid_label in uid_labels])
                    ):
                        raise Exception(
                            ValueError,
                            f"Not all uid_labels '{uid_labels}' in file: {fname}",
                        )
                    # get the column index for uid fields
                    indices_uids = [
                        i for i, field in enumerate(column_names) if field in uid_labels
                    ]
                    try:
                        index_raw = column_names.index(raw_label)
                    except ValueError as e:
                        raise Exception(
                            e, f"Raw label '{raw_label}' not in file: {fname}"
                        )
                    for row in reader:
                        key = tuple([infer_type(row[i]) for i in indices_uids])
                        value = row[index_raw]
                        mapping[key] = value
        return mapping
