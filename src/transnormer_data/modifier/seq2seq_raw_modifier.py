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
        self.raw_trg = layer
        self.tok_trg = f"{layer}_tok"
        self.tok_src = "orig_tok" if self.raw_trg == "norm" else "norm_tok"
        self.ws_trg = f"{layer}_ws"
        self.spans_trg = f"{layer}_spans"
        self.alignment = "alignment"

        # Score in range {0, 1} that tells us the proportion of language
        # detection models that guessed that the sample is in German
        self.lang_de_score = "lang_de"

        # Seq2seq model
        logger.info(f'Loading model "{model_name}"')
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self._device
        )

        # Setting
        self.recompute_alignments = recompute_alignments

    def modify_batch(self, batch: Dict[str, List]):
        """
        TODO
        """

        original_raw = batch[self.raw_trg]

        inputs = self.tokenizer(
            # TODO lowercase is only for caser
            [string.lower() for string in batch[self.raw_trg]],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch[self.raw_trg] = output_str

        # TODO: untested
        # Propagate changes
        # Convert batch (dict of lists) to samples (dict) and back to batch
        keys = batch.keys()
        list_length = len(next(iter(batch.values())))
        batch_updated: Dict[str, List] = {key: [] for key in keys}
        for i in range(list_length):
            sample = {key: batch[key][i] for key in keys}
            # ignore non-German
            if self.lang_de_score in sample:
                if sample[self.lang_de_score] == 0:
                    sample[self.raw_trg] = original_raw[i]
            sample = self.update_rest_of_sample(sample)
            # Convert sample back to batch
            for key in keys:
                batch_updated[key].append(sample[key])

        return batch

    def update_rest_of_sample(self, sample: Dict):

        self.update_raw_from_tok(
            sample,
            key_raw=self.raw_trg,
            key_tok=self.tok_trg,
        )
        self.update_spans_and_ws_from_tok_and_raw(
            sample,
            key_tokens=self.tok_trg,
            key_raw=self.raw_trg,
            key_ws=self.ws_trg,
            key_spans=self.spans_trg,
        )
        if self.recompute_alignments:
            self.update_alignment(
                sample,
                key_tokens_src=self.tok_src,
                key_tokens_trg=self.tok_trg,
                key_alignment=self.alignment,
            )

        return sample

    # dummy
    def modify_sample(self, sample):
        pass
