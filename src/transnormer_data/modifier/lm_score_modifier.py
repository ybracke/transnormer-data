import os

from typing import Dict, Optional, Set, Union

import logging
import datasets
import torch
import transformers
import numpy as np

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data import utils

logger = logging.getLogger(__name__)


class LMScorer(object):
    def __init__(self, model_name: str):

        logger.info(f'Loading huggingface language model "{model_name}"')
        # TODO: catch if model does not exist
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(
            device
        )
        self.model.eval()
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model_name = model_name

        return

    def __call__(self, text: str) -> Dict[str, float]:
        """
        Returns a dictionary of the language model score

        Possible output:
        {
        "dbmdz/german-gpt2" : -5.6789
        }
        """
        scores = dict()
        scores[self.model_name] = self.predict_logprobs(text)
        return scores

    def predict_logprobs(self, input_sample: str) -> float:
        # Tokenize the input sentence
        inputs = self.tokenizer(input_sample, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"].to(self.model.device)

        # Get the output logits from the model
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits

        # Shift the logits and input_ids to align the predictions with the true values
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Flatten the tokens and logits for computing the loss
        # Compute the log probabilities
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        log_probs = -loss.view(shift_labels.size())

        # Aggregate sentence score
        score = log_probs.mean(-1)

        return score.cpu().detach().numpy()[0]


class LMScoreModifier(BaseDatasetModifier):
    def __init__(
        self, layer: Optional[str] = None, language_model: Optional[str] = None
    ) -> None:
        """
        This modifier runs language tool over the raw version of the source or target layer of the corpus and adds language models probibility score as an additional property to the dataset.

        The default layer that language detection is applied to is "norm".
        """

        # Set layer
        accepted_layers = {"orig", "norm"}
        layer = "norm" if layer is None else layer
        if layer not in accepted_layers:
            raise NotImplementedError(
                f"""LMScoreModifier is not implemented for layer '{layer}'. Choose one of {accepted_layers}"""
            )
        self.raw = layer

        # LM
        model_name = "dbmdz/german-gpt2" if language_model is None else language_model
        self.lm_scorer = LMScorer(model_name)

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Add language model score as a property to the sample.
        Score is the log probability times -1 (i.e. a positive float).
        """

        scores = self.lm_scorer(sample[self.raw])
        # Rounding only works with positive values
        # Thus, lower value means more probable sentence
        scores_pos_rnd = {
            model: np.round(score * -1, 4) for (model, score) in scores.items()
        }
        sample.update(scores_pos_rnd)
        return sample
