import logging
from typing import Optional

import transformers

from transnormer_data.base_dataset_modifier import BaseDatasetModifier

logger = logging.getLogger(__name__)


class Seq2SeqRawModifier(BaseDatasetModifier):
    def __init__(
        self,
        layer: str = "norm",
        model: Optional[str] = None,
        tokenizer: Optional[str] = None,
        recompute_alignments: bool = True,
    ) -> None:
        """
        Modifier for applying sequence-to-sequence models to raw text

        Modifies the raw text version on the target layer via a seq2seq model
        and propagates the changes to the tokenized version of the layer, and
        if necessary, recomputes alignments.
        """

    def modify_batch(self, batch_size):
        pass

    def modify_dataset(self, dataset):
        pass
