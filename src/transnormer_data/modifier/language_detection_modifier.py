import os
from typing import Dict, Optional

import cld3
import fasttext
from py3langid.langid import MODEL_FILE, LanguageIdentifier

from transnormer_data.base_dataset_modifier import BaseDatasetModifier

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
        "lang_fastText" : "de",
        "lang_py3langid" : "en",
        "lang_cld3" : "de",
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
    def __init__(self, layer: Optional[str] = None) -> None:
        """
        This modifier runs language detection algorithms over the raw version of the source or target layer of the corpus and adds the language labels as additional properties to the dataset.

        The default layer that language detection is applied to is "orig".
        """

        self.languagedetector = LanguageIdentificationEnsemble()

        # Set layer
        accepted_layers = {"orig", "norm"}
        if layer is None:
            layer = "orig"
        if layer not in accepted_layers:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for layer '{layer}'. Choose one of {accepted_layers}"""
            )
        self.raw = layer

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.
        """

        guesses = self.languagedetector(sample[self.raw])
        sample.update(guesses)
        sample["lang_de"] = round(
            sum(lang == "de" for lang in guesses.values()) / len(guesses), 3
        )

        return sample
