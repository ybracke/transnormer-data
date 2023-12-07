import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import datasets

from transnormer_data import utils
from transnormer_data.maker.dta_maker import DtaMaker

# from transnormer_data.modifier.vanilla_dtak_modifier import VanillaDtakModifier # TODO


class DtakMaker(DtaMaker):
    """An object that creates a dataset in the transnormer format from the DTA Eval Corpus in its original XML format, plus metadata in JSONL format"""

    def __init__(
        self,
        path_data: Union[str, os.PathLike],
        path_metadata: Union[str, os.PathLike],
        path_output: Union[str, os.PathLike],
    ) -> None:
        """Initialize the maker with paths to the data files, metadata file and output directory"""
        super().__init__(path_data, path_metadata, path_output)

        # self._modifier: Optional[VanillaDtakModifier] = None # TODO

    def make(self):
        return 

    def _load_data(self) -> datasets.Dataset:
        """
        Reads data from a DTA ddctabs file into a dataset

        Input format documentation: https://kaskade.dwds.de/~moocow/software/ddc/ddc_tabs.html
        """
        basenames = []
        sents_orig_tok: List[List[str]] = []
        sents_orig_xlit: List[List[str]] = []
        sents_orig_lemma: List[List[str]] = []
        sents_orig_pos: List[List[str]] = []
        sents_orig_ws: List[List[bool]] = []
        sents_norm_tok: List[List[str]] = []
        par_idxs = []
        for fname_in in glob.iglob(os.path.join(self.path_data, "*"), recursive=True):

            basename = utils.get_basename_no_ext(fname_in)
            par_idx = 0 # reset paragraph index for every document

            columns = {}  # column tab index
            attrs = []  # tabs per line
            in_s = False  # within sentence (= not first token of sentence)

            # initialize single sent lists
            sent_orig_tok: List[str] = []
            sent_orig_xlit: List[str] = []
            sent_orig_lemma: List[str] = []
            sent_orig_pos: List[str] = []
            sent_orig_ws: List[bool] = []
            sent_norm_tok: List[str] = []

            with open(fname_in, "r", encoding="utf8") as fh:
                for line in fh:
                    line = line.strip()

                    # (A) Metadata line
                    if line.startswith("%%$DDC:index["):
                        match = re.split(" |=", line.strip())
                        try:
                            i = int(re.search(r"\d+", match[0])[0])
                        except:
                            raise Exception(
                                "Couldn't parse ddc-tabs input file correctly."
                            )
                        long = match[1]
                        short = match[2]
                        if long == "Token" or short == "w":
                            columns["xlit"] = i
                        elif long == "Utf8" or short == "u":
                            columns["original"] = i
                        elif long == "CanonicalToken" or short == "v":
                            columns["normalized"] = i
                        elif long == "Pos" or short == "p":
                            columns["pos"] = i
                        elif long == "Lemma" or short == "l":
                            columns["lemma"] = i
                        elif long == "WordSept" or short == "ws":
                            columns["ws"] = i

                    # (B) Token line: inside sentence
                    elif not line.startswith("%%") and line:

                        # 1. Get tokens/annotations 
                        attrs = line.split("\t")
                        orig = attrs[columns["original"]]
                        orig_xlit = attrs[columns["xlit"]]
                        orig_lemma = attrs[columns["lemma"]]
                        orig_pos = attrs[columns["pos"]]
                        orig_ws = (
                            bool(int(attrs[columns["ws"]])) if in_s else False
                        )  # always false for first token in sentence
                        norm = attrs[columns["normalized"]]

                        # 2. Default modification: Replace the pre-normalized punctuation  # on norm layer with the original unnormalized punctuation
                        # Look at POS-annotations for that
                        if orig_pos.startswith("$"): # STTS-Tags "$,", "$.", "$("
                            norm = orig

                        # 3. Add to sentence list
                        sent_orig_tok.append(orig)
                        sent_orig_xlit.append(orig_xlit)
                        sent_orig_lemma.append(orig_lemma)
                        sent_orig_pos.append(orig_pos)
                        sent_orig_ws.append(orig_ws)

                        # 4. Split norm token at "_"
                        norm, _ = self.custom_split(norm)
                        sent_norm_tok.extend(norm)

                        # Set in-sentence flag
                        in_s = True

                    # (C) Empty line: end of sentence
                    elif line.strip() == "":
                        # Append single sentences to sentences lists
                        par_idxs.append(par_idx)
                        basenames.append(basename)
                        sents_orig_tok.append(sent_orig_tok)
                        sents_orig_xlit.append(sent_orig_xlit)
                        sents_orig_lemma.append(sent_orig_lemma)
                        sents_orig_pos.append(sent_orig_pos)
                        sents_orig_ws.append(sent_orig_ws)
                        sents_norm_tok.append(sent_norm_tok)

                        # Reset
                        in_s = False
                        par_idx += 1
                        sent_orig_tok = []
                        sent_orig_xlit = []
                        sent_orig_lemma = []
                        sent_orig_pos = []
                        sent_orig_ws = []
                        sent_norm_tok = []

        length = len(basenames)
        assert length == len(par_idxs) == len(sents_orig_tok) == len(sents_orig_ws) == len(sents_norm_tok)

        return datasets.Dataset.from_dict(
            {
                "basename": basenames,
                "par_idx": par_idxs,
                "orig_tok": sents_orig_tok,
                "orig_xlit" : sents_orig_xlit, 
                "orig_lemma" : sents_orig_lemma, 
                "orig_pos" : sents_orig_pos, 
                "orig_ws": sents_orig_ws,
                "norm_tok": sents_norm_tok,
                "norm_ws": [None for i in range(length)],
            }
        )

    @staticmethod
    def custom_split(input_string: str) -> Tuple[List[str], bool]:
        """Returns the list of token(s) and True if there actually was a split"""
        if "_" in input_string:
            return re.split(r"_", input_string), True
        else:
            return [input_string], False

