import logging
from typing import Dict, List, Optional

import torch
import transformers

from transnormer_data.base_dataset_modifier import BaseDatasetModifier

logger = logging.getLogger(__name__)


class Seq2SeqRawModifier(BaseDatasetModifier):
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer: Optional[str] = None,
        layer: str = "norm",
        recompute_alignments: bool = True,
    ) -> None:
        """
        Modifier for applying sequence-to-sequence models to raw text

        Modifies the raw text version on the target layer via a seq2seq model
        and propagates the changes to the tokenized version of the layer, and
        if necessary, recomputes alignments.
        """

        # Set layer
        accepted_layers = {"orig", "norm"}
        layer = "norm" if layer is None else layer
        if layer not in accepted_layers:
            raise NotImplementedError(
                f"""LMScoreModifier is not implemented for layer '{layer}'. Choose one of {accepted_layers}"""
            )
        self.raw = layer
        self.tok = f"{layer}_tok"
        self.ws = f"{layer}_ws"
        self.spans = f"{layer}_spans"
        self.alignment = "alignment"

        # Score in range {0, 1} that tells us the proportion of language
        # detection models that guessed that the sample is in German
        self.lang_de_score = "lang_de"

        # Seq2seq model
        logger.info(f'Loading model "{model_name}"')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            device
        )

        # Setting
        self.recompute_alignments = recompute_alignments

    def modify_batch(self, batch: Dict[str, List]):

        # layer_layer = f"{self.raw}_lower"
        # batch[layer_layer] = [string.lower() for string in batch["norm"]]
        inputs = self.tokenizer(
            # batch[layer_layer],
            [string.lower() for string in batch["norm"]],
            return_tensors="pt",
        )
        outputs = self.model.generate(**inputs)
        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch[self.raw] = output_str
        # batch.pop(layer_layer)

        # TODO
        if self.recompute_alignments:
            pass

        return batch

    # dummy
    def modify_sample(self, sample):
        pass
