# import json
import os
import glob
from typing import Dict, List, Optional, Tuple, Union

import datasets
from lxml import etree

from transnormer_data import utils
from transnormer_data.base_maker import BaseMaker

# from transnormer_data.modifier.vanilla_dta_modifier import VanillaDtaModifier


class RidgesMaker(BaseMaker):
    """An object that creates a dataset in the transnormer format from the RIDGES corpus in its original Paula XML format

    TODO: metadata
    """

    def __init__(
        self,
        path_data_orig: Union[str, os.PathLike],
        path_data_norm: Union[str, os.PathLike],
        # path_metadata: Union[str, os.PathLike],
        path_output: Union[str, os.PathLike],
    ) -> None:
        """Initialize the maker with paths to the data files, metadata file and output directory"""

        self.path_data_orig = path_data_orig
        self.path_data_norm = path_data_norm
        # self.path_metadata = path_metadata
        self.path_output = path_output

        self._dataset: Optional[datasets.Dataset] = None
        self._metadata: Optional[Dict[str, Dict]] = None

        # self._modifier: Optional[BaseDatasetModifier] = None # TODO

    def make(self, save: bool = False) -> datasets.Dataset:
        """Create a datasets.Dataset object from the paths passed to the constructor.

        Pass `save=True` to save the dataset in JSONL format to the output directory that was passed to the constructor. If the directory does not exists, it will be created.
        """
        # self._metadata = self._load_metadata() # TODO
        self._dataset = self._load_data()
        # TODO: Jetzt sollten die Daten aussehen wie in testdata/jsonl/ridges/simplest.jsonl

        # self._dataset = self._join_data_and_metadata(join_on="basename")
        # TODO
        # self._modifier = VanillaDtaModifier()
        # self._dataset = self._modifier.modify_dataset(self._dataset)
        # TODO: Jetzt sollten die Daten aussehen wie in testdata/jsonl/ridges/tokenized.jsonl
        if save:
            if not os.path.isdir(self.path_output):
                os.makedirs(self.path_output)
            utils.save_dataset_to_json_grouped_by_property(
                self._dataset, property="basename", path_outdir=self.path_output
            )
        return self._dataset

    def _load_data(self) -> datasets.Dataset:
        """
        Reads from a DTA EvalCorpus XML file into a dataset TODO

        """
        basenames = []
        sents_orig_all = []
        sents_norm_all = []
        par_idxs_all = []
        files_orig = sorted(glob.glob(self.path_data_orig))
        files_norm = sorted(glob.glob(self.path_data_norm))
        # Assert file names in the data directories match
        assert [utils.get_basename_no_ext(f) for f in files_orig] == [
            utils.get_basename_no_ext(f) for f in files_norm
        ]
        for file_orig, file_norm in zip(files_orig, files_norm):
            basename = utils.get_basename_no_ext(file_orig)
            tree_orig = etree.parse(file_orig)
            tree_norm = etree.parse(file_norm)
            bodytext_orig = tree_orig.find("//body").text
            bodytext_norm = tree_norm.find("//body").text
            
            # Get sentences
            # TODO: This removes the sentence final dot and never puts it back
            sents_orig = bodytext_orig.split(" . ")
            sents_norm = bodytext_norm.split(" . ")
            # Assert same number of sentences per file version
            # We assume that sentences with the same index belong together
            if len(sents_orig) != len(sents_norm): 
                continue

            else:
                # Extend overall lists
                sents_orig_all.extend(sents_orig)
                sents_norm_all.extend(sents_norm)
                basenames.extend([basename for i in range(len(sents_orig))])
                par_idxs_all.extend(list(range(len(sents_orig))))

        length = len(basenames)
        assert length == len(par_idxs_all) == len(sents_orig_all) == len(sents_norm_all)
        # print(length, len(par_idxs_all), len(sents_orig_all), len(sents_norm_all))
        return datasets.Dataset.from_dict(
            {
                "basename": basenames,
                "par_idx": par_idxs_all,
                "orig": sents_orig_all,
                "norm": sents_norm_all,
            }
        )
