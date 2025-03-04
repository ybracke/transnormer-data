import logging
from typing import Dict, Optional

import numpy as np
import torch
import transformers

from transnormer_data.base_dataset_modifier import BaseDatasetModifier

logger = logging.getLogger(__name__)


class LMScorer(object):
    def __init__(self, model_name: str, prefix: Optional[str] = None):
        logger.info(f'Loading huggingface language model "{model_name}"')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(
            device
        )
        self.model.eval()
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.key = f"{prefix}_{model_name}" if prefix else f"{model_name}"

        return

    def __call__(self, text: str) -> Dict[str, np.float32]:
        """
        Returns a dictionary of the language model score (negative log likelihood)

        Possible output:
        {
        "norm_dbmdz/german-gpt2" : 5.6789
        }
        """
        scores = dict()
        scores[f"{self.key}"] = self.predict_logprobs(text)
        return scores

    def predict_logprobs(self, input_str: str) -> float:
        # Tokenize the input sentence
        inputs = self.tokenizer(
            input_str,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)
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
        neg_log_probs = loss.view(shift_labels.size())

        # Aggregate sentence score
        score = neg_log_probs.mean(-1)

        return score.cpu().detach().numpy()[0]


class LMScoreModifier(BaseDatasetModifier):
    def __init__(
        self, layer: Optional[str] = None, language_model: Optional[str] = None
    ) -> None:
        """
        Modifier that adds a language model (LM) probability score to each record.

        LM must be a huggingface model, default is "dbmdz/german-gpt2".
        By default, LM scores are computed for the layer "norm".

        A new property is created for the score with the language model's name, e.g.

        ```json
        {
            "dbmdz/german-gpt2" : 5.6789
        }
        ```
        """

        # Set layer
        layer = "norm" if layer is None else layer
        self.raw = layer

        # LM
        model_name = "dbmdz/german-gpt2" if language_model is None else language_model
        self.lm_scorer = LMScorer(model_name, prefix=layer)

    def modify_sample(self, sample: Dict) -> Dict:
        """
        Add language model score as a property to the sample.

        Score is the negative log likelihood
        """

        scores = self.lm_scorer(sample[self.raw])
        # convert np.float32 to float
        scores_float = {
            key: score.item()
            for (key, score) in scores.items()
            # round to 10 decimal points
            # key: round(score.item(),4) for (key, score) in scores.items()
        }
        sample.update(scores_float)
        return sample
