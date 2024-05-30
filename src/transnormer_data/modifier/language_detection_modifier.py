import os

from typing import Dict, Optional, Set, Union

import datasets
import spacy
import cld3
from py3langid.langid import LanguageIdentifier, MODEL_FILE
import fasttext

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data import utils

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
)
MODELPATH_FT = os.path.join(ROOT, "resources/lid.176.ftz")


class LanguageIdentificationEnsemble(object):
    def __init__(self):
        self.model_ft = fasttext.FastText._FastText(model_path=MODELPATH_FT)
        self.model_li = LanguageIdentifier.from_pickled_model(
            MODEL_FILE, norm_probs=True
        )
        return

    def __call__(self, text: str) -> Dict[str, str]:
        """
        Returns what each classifier in the ensemble holds to be the most probable language

        Possible output:
        {
        "fastText" : "de",
        "py3langid" : "en",
        "cld3" : "de",
        }
        """
        labels = dict()
        # fastText
        langs_ft, _ = self.model_ft.predict(text)
        top_label = langs_ft[0]  # e.g. '__label__de'
        labels["lang_fastText"] = top_label[-2:]
        # py3langid
        lang_li, _ = self.model_li.classify(text)
        labels["lang_py3langid"] = lang_li
        # cld3
        labels["lang_cld3"] = cld3.get_language(text).language
        return labels


class LanguageDetectionModifier(BaseDatasetModifier):
    def __init__(self) -> None:
        """
        This modifier runs language detection algorithms over the raw version of the source or target layer of the corpus and adds the language labels as additional properties to the dataset.

        The default layer that language detection is applied to is "orig".
        """

        self.languagedetector = LanguageIdentificationEnsemble()

        # Relevant keys in the sample dictionary
        self.raw = "orig"

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.
        """

        guesses = self.languagedetector(sample[self.raw])
        sample.update(guesses)

        return sample


    def modify_dataset(
        self,
        dataset: datasets.Dataset,
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Union[datasets.Dataset, None]:
        dataset = dataset.map(self.modify_sample)
        if save_to:
            if not os.path.isdir(save_to):
                os.makedirs(save_to)
            utils.save_dataset_to_json_grouped_by_property(
                dataset, property="basename", path_outdir=save_to
            )
        return dataset