import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import datasets
import dotenv
import openai
import tiktoken
from openai.types.chat.chat_completion import ChatCompletion
import tiktoken

from transnormer_data import utils
from transnormer_data.base_dataset_modifier import BaseDatasetModifier

dotenv.load_dotenv()

# TODO
# logging settings
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=".log/gpt_modifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)

# TODO: fix seed, temperature, ...


# Simplified version of: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken#6-counting-tokens-for-chat-completions-api-calls
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0125",
    }:
        tokens_per_message = 3
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o" in model:
        print(
            "Warning: gpt-4o may update over time. Returning num tokens assuming gpt-4o-2024-05-13"
        )
        return num_tokens_from_messages(messages, model="gpt-4o-2024-05-13")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tiktokens(s: str, model) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(s))


class GPTModifier(BaseDatasetModifier):
    def __init__(
        self,
        model_name: str,
        user_prompt: str,
        system_prompt: str = "",
        example_query: str = "",
        example_response: str = "",
        max_len_prompt: Optional[int] = None,
    ) -> None:
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

        self.model = model_name
        # maximum input length in tiktokens
        self.max_len_prompt = max_len_prompt  # TODO: if None: take model's max

        # base prompt
        self.prompt_base = self._build_prompt_base(
            user_prompt, system_prompt, example_query, example_response
        )

    def record_to_example_line(self, record: Dict[str, Any]) -> str:
        # Future TODO: offer different choices (add only 'orig', add both 'orig' and 'norm', add '*_tok', etc.)
        return f"{record['orig']}\t{record['norm']}"
        # return f"{record['norm']}"

    def complete_prompt(
        self, prompt: List[Dict[str, str]], examples: List[str]
    ) -> List[Dict[str, str]]:
        """
        Add line to the prompt.
        """
        prompt[-1]["content"] = "\n\n".join(examples)
        # HOTFIX
        # prompt[-1]["content"] += "\n\n" + "\n\n".join(examples) + "\n\n##assistant:\n\n"
        return prompt

    def _build_prompt_base(
        self,
        user_prompt: str,
        system_prompt: str = "",
        example_query: str = "",
        example_response: str = "",
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt},
            )
        messages.append(
            {"role": "user", "content": user_prompt},
        )
        if example_query and example_response:
            messages.append(
                {"role": "user", "content": example_query},
            )
            messages.append(
                {"role": "assistant", "content": example_response},
            )
        messages.append({"role": "user", "content": ""})
        return messages

    def parse_response(self, response: Optional[ChatCompletion]) -> List[str]:
        """
        Parse the response of the OpenAI client into a list of strings (= sentences)
        """
        if response is not None:
            n_tokens = response.usage.total_tokens  # type: ignore
            logger.info(f"{n_tokens} tokens used.")
            answer = response.choices[0].message.content
        else:
            answers = ["###FAIL###"]
            answer = None
        if answer is not None:
            answers = answer.strip().split("\n\n")
        else:
            answers = ["###FAIL###"]
        return answers

    def query_client(self, messages: List[Dict[str, str]]) -> Optional[ChatCompletion]:
        """
        Query the OpenAI API with `messages`.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
            )
            time.sleep(3.0)  # prevent rate limit errors
            return response
        except Exception as err:
            logger.error(err)
        return None

    def modify_dataset(
        self,
        dataset: datasets.Dataset,
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Union[datasets.Dataset, None]:
        """
        Modify dataset by querying the API with a prompt.

        The loop makes sure the prompt length does not exceed the max_len_prompt
        """
        preds: List[str] = []
        prompt = self.prompt_base
        len_prompt_base = num_tokens_from_messages(self.prompt_base)
        examples: List[str] = []
        len_examples = 0
        for record in dataset:
            current_example = self.record_to_example_line(record)
            len_current_example = num_tiktokens(current_example, model=self.model)
            len_total = len_prompt_base + len_examples + len_current_example
            if len_total > self.max_len_prompt:
                # build prompt and pass to client without current example
                prompt = self.complete_prompt(prompt, examples)
                response = self.query_client(prompt)
                answers = self.parse_response(response)
                # collect predictions
                preds.extend([{"pred": pred} for pred in answers])
                # reset example list and length
                examples = [current_example]
                len_examples = len_current_example
                len_total = len_prompt_base + len_examples + len_current_example
            else:
                # extend example list and length
                examples.append(current_example)
                len_examples += len_current_example

        # If (for some reason) the number of preds does not match the number of examples in dataset, we still want to store the preds somewhere so that they are not lost
        if len(preds) != len(dataset):
            print(preds)  # TODO: implement a better solution than printing
            ds_final = None
        else:
            # convert preds to dataset
            preds_ds = datasets.Dataset.from_list(preds)
            ds_final = datasets.concatenate_datasets([dataset, preds_ds], axis=1)
            print(ds_final[:])

        if save_to and ds_final is not None:
            if not os.path.isdir(save_to):
                os.makedirs(save_to)
            try:
                utils.save_dataset_to_json_grouped_by_property(
                    ds_final, property="basename", path_outdir=save_to
                )
            except:
                print(preds)

        return ds_final
