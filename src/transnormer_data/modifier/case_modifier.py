import logging
from typing import Dict, List, Optional

from transnormer_data.modifier.seq2seq_raw_modifier import Seq2SeqRawModifier

logger = logging.getLogger(__name__)


class CaseModifier(Seq2SeqRawModifier):
    def __init__(
        self,
        model_name: Optional[str] = None,
        layer: str = "norm",
    ) -> None:
        """
        Modifier for applying sequence-to-sequence casing models to raw text

        Modifies the raw text version on the target layer via a seq2seq model
        and propagates the changes to the tokenized version of the layer. Since
        a caser should not change the tokenization, the alignments between
        source and target layer do not have to be recomputed.
        """

        return super().__init__(model_name, layer, recompute_alignments=False)

    def modify_batch(self, batch: Dict[str, List]):
        """ """
        # keep original text for assert
        old_raw = batch[self.raw_trg]
        # lowercase for caser
        batch[self.raw_trg] = [string.lower() for string in batch[self.raw_trg]]
        batch = super().modify_batch(batch)
        # TODO: assert that no changes have occured other than casing
        # TODO: what to do if other changes occured?
        for new, old in zip(batch[self.raw_trg], old_raw):
            assert new.lower() == old.lower()
            # logger.warning() # TODO
        return batch
