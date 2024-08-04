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
        """ """

        # Keep previous raw text as backup
        raw_before: List[str] = batch[self.raw_trg]
        # Lowercased version of original
        raw_before_lc = [string.lower() for string in batch[self.raw_trg]]

        inputs = self.tokenizer(
            raw_before_lc,
            batch[self.raw_trg],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=999_999)

        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch[self.raw_trg] = output_str

        # Propagate changes
        # Convert batch (dict of lists) to samples (dict) and back to batch
        keys = batch.keys()
        list_length = len(next(iter(batch.values())))
        batch_updated: Dict[str, List] = {key: [] for key in keys}
        for i in range(list_length):
            sample = {key: batch[key][i] for key in keys}
            # Ignore non-German samples
            if self.lang_de_score in sample:
                if sample[self.lang_de_score] == 0:
                    sample[self.raw_trg] = raw_before[i]
            # Catch cases where caser did more than it is supposed to do
            # TODO: IDs should not be hard-coded
            elif raw_before_lc[i] != batch[self.raw_trg][i].lower():
                logger.warning(
                    f"Warning: Caser changed more than case. Will ignore caser output and keep sample in previous state. ID: ({batch['basename'][i]}, {batch['par_idx'][i]})."
                )
                sample[self.raw_trg] = raw_before[i]
            # If everything is okay
            else:
                sample = self.update_rest_of_sample(sample, self.recompute_alignments)
            # Convert sample back to batch
            for key in keys:
                batch_updated[key].append(sample[key])

        return batch_updated
