import logging
from typing import Dict, List, Optional

import torch

from transnormer_data.modifier.seq2seq_raw_modifier import Seq2SeqRawModifier

logger = logging.getLogger(__name__)


class CaseModifier(Seq2SeqRawModifier):
    def __init__(
        self,
        layer: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Modifier for applying sequence-to-sequence casing models to raw text

        Modifies the raw text version on the target layer via a seq2seq model
        and propagates the changes to the tokenized version of the layer. Since
        a caser should not change the tokenization, the alignments between
        source and target layer do not have to be recomputed.
        """
        return super().__init__(layer, model_name, recompute_alignments=False)

    def modify_batch(self, batch: Dict[str, List]):
        # Keep previous raw text as backup
        raw_before: List[str] = batch[self.raw_trg]
        # Lowercased version of original
        raw_before_lc = [string.lower() for string in batch[self.raw_trg]]

        inputs = self.tokenizer(
            raw_before_lc,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.gen_cfg)

        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch[self.raw_trg] = [s.strip() for s in output_str]

        # Propagate changes
        # Convert batch (dict of lists) to samples (dict) and back to batch
        keys = batch.keys()
        list_length = len(next(iter(batch.values())))
        batch_updated: Dict[str, List] = {key: [] for key in keys}
        for i in range(list_length):

            sample = {key: batch[key][i] for key in keys}
            # Ignore non-German samples
            if self.lang_de_score in sample and sample[self.lang_de_score] == 0:
                sample[self.raw_trg] = raw_before[i]
            # Check whether caser changed more than it is supposed to
            # HOTFIX: allow spacing differences
            elif raw_before_lc[i].replace(" ", "") != batch[self.raw_trg][
                i
            ].lower().replace(" ", ""):
                # TODO: IDs should not be hard-coded
                logger.warning(
                    f"Caser changed more than case. Will ignore caser output and keep sample in previous state. ID: ({batch['basename'][i]}, {batch['par_idx'][i]})."
                )
                logger.warning(f"Generated: '{batch[self.raw_trg][i]}'")
                sample[self.raw_trg] = raw_before[i]
            # If everything is okay
            else:
                sample = self.update_rest_of_sample(sample, self.recompute_alignments)
            # Convert sample back to batch
            for key in keys:
                batch_updated[key].append(sample[key])

        return batch_updated
