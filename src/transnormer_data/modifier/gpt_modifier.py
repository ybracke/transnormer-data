import logging
import os
import time

from typing import Dict, List, Optional, Set, Union

import datasets
import openai
import dotenv

from transnormer_data.base_dataset_modifier import BaseDatasetModifier
from transnormer_data import utils


dotenv.load_dotenv()

# logging settings
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=".log/gpt_modifier.log",  # TODO
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)

# TODO: fix seed, temperature, ...


class GPTModifier(BaseDatasetModifier):
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, example_query: str, example_response: str) -> None:
        """
        This modifier uses OpenAI's GPT models (via API) to apply changes to a layer of the data and propagates the changes to the other layers.

        TODO: Fixed to a specific layer?
        """

        # Keys in the sample dictionary
        self.raw = "norm"
        self.tok = "norm_tok"
        self.ws = "norm_ws"
        self.spans = "norm_spans"
        self.tok_src = "orig_tok"
        self.alignment = "alignment"

        # GPT setup
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # TODO: configuration file with gpt model
        # TODO: prompt file for system prompt and example_query and response
        self.model = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.example_query = example_query
        self.example_response = example_response

    def modify_dataset(
        self,
        dataset: datasets.Dataset,
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Union[datasets.Dataset, None]:
        # TODO: Anders als bei den anderen Modifiern sollte hier nicht jedes
        # Beispiel einzeln behandelt werden, sondern immer so viele Beispiele wie möglich. Eine `modify_sample`-Funktion, die ein einzelnes Beispiel behandelt brauchen wir also nicht in der Form, sondern stattdessen eine die mehrere behandelt (e.g. `modify_samples`)

        batch_size = 4
        dataset = dataset.map(self.modify_samples, batch_size=batch_size, fn_kwargs={"batch_size": batch_size}) # TODO: fixed batch size

        # if save_to:
        #     if not os.path.isdir(save_to):
        #         os.makedirs(save_to)
        #     utils.save_dataset_to_json_grouped_by_property(
        #         dataset, property="basename", path_outdir=save_to
        #     )
        return dataset

    def modify_samples(self, samples: Dict, batch_size: int) -> List[Dict]:
        sample_sents = [sample["norm"] for sample in samples] # TODO: this shouldn't be fixed here
        # TODO: this is a dummy
        answers: List[str] = self.query_client_mockup(sample_sents)
        samples = self._add_annotation(answers, samples)
        return samples

    def _add_annotation(self, gpt_annos: List[str], samples: List[Dict]):
        """ Adds the annotations provided by GPT to a list of samples. 
        
        Assumes that the elements in `gpt_annos` belong to the elements in `samples`, and that they are in the same order. """

        samples_updated = []
        for sample, gpt_anno in zip(samples, gpt_annos):
            sample["gpt_anno"] = gpt_anno
            samples_updated.append(sample)
        return samples_updated

    def query_client_mockup(
        self,
        samples: List[str],
    ) -> List[str]:
        # return same-length list of strings
        return ["FOO" for s in samples] 

    def query_client(
        self,
        samples: List[str],
    ) -> List[str]:
        """Query the OpenAI API to generate a response for a list of sentences."""
        messages = self._build_messages(
            samples,
            self.system_prompt,
            self.user_prompt,
            self.example_query,
            self.example_response,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
            )
            answer = response.choices[0].message.content
            if response is not None: 
                # if answer is not None:
                    # answers.append(answer.strip())
                # else:
                #     answers.append("###FAIL###")
                n_tokens = response.usage.total_tokens # type: ignore
                logger.info(f"{n_tokens} tokens used.")
            else:
                answers.append("###FAIL###")
            time.sleep(3.0)  # prevent rate limit errors
        except Exception as err:
            logger.error(err)
        answers = answer.split("\n\n")
        return answers

    def _build_messages(
        self,
        samples: List[str],
        system_prompt: str,
        user_prompt: str,
        example_query: Optional[str]=None,
        example_response: Optional[str]=None,
    ) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if example_query is not None and example_response is not None:
            messages.append(
                {"role": "user", "content": example_query},
            )
            messages.append(
                {"role": "assistant", "content": example_response},
            )
        messages.append({"role": "user", "content": "\n\n".join(samples)})
        return messages



'''
    def modify_sample(self, sample: Dict) -> Dict:
        """
        Apply a modification function to a property of the sample
        and propagate the modifications to other properties of the sample.

        Here, the modification is applied to norm_raw (via LanguageTool) and
        the changes are propagated to norm_tok, etc.
        """

        # Update raw via LanguageTool
        raw_old = sample[self.raw]
        raw_new = self.langtool.correct(raw_old)
        any_changes = raw_new != raw_old
        if any_changes:
            sample[self.raw] = raw_new
            self.update_tok_from_raw(
                sample, key_raw=self.raw, key_tok=self.tok, key_ws=self.ws
            )

            # Update spans
            self.update_spans_and_ws_from_tok_and_raw(
                sample,
                key_tokens=self.tok,
                key_raw=self.raw,
                key_ws=self.ws,
                key_spans=self.spans,
            )

            # Update alignment
            self.update_alignment(
                sample,
                key_tokens_src=self.tok_src,
                key_tokens_trg=self.tok,
                key_alignment=self.alignment,
            )
        return sample

'''
