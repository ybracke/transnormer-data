from datetime import datetime
import logging
import unittest
import pytest

from typing import List

import datasets

from openai.types.completion_usage import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

from transnormer_data.modifier.gpt_modifier import (
    GPTModifier,
    num_tokens_from_messages,
    num_tiktokens,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=".log/test_gptmodifier.log",  # TODO
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)


class GPTModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = GPTModifier(
            "gpt-3.5-turbo-0125",
            "You are a helpful assistant.",
            "Do nothing.",
            "",
            "",
            400,
        )  # TODO dummy vars
        self.mock_response_01 = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Erster Satz.\n\nZweiter Satz.",
                        role="assistant",
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage=CompletionUsage(
                completion_tokens=10, prompt_tokens=10, total_tokens=10
            ),
        )

    def tearDown(self) -> None:
        pass

    def test_build_prompt_base(self) -> None:
        system_prompt = "A"
        user_prompt = "B"
        example_query = "C"
        example_response = "D"
        target_prompt = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
            {"role": "user", "content": ""},
        ]
        prompt = self.modifier._build_prompt_base(
            system_prompt, user_prompt, example_query, example_response
        )
        assert prompt == target_prompt

    def test_num_tokens_from_message(self) -> None:
        target_prompt = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
            {"role": "user", "content": "E\tF\n\nG\tH"},
        ]
        len_target_prompt = 32
        system_prompt = "A"
        user_prompt = "B"
        example_query = "C"
        example_response = "D"
        messages = self.modifier._build_prompt_base(
            system_prompt,
            user_prompt,
            example_query,
            example_response,
        )
        examples = ["E\tF", "G\tH"]
        messages = self.modifier.complete_prompt(messages, examples)
        assert messages == target_prompt
        num_tokens = num_tokens_from_messages(messages)
        assert num_tokens == len_target_prompt

    def test_parse_response(self) -> None:
        # mock
        completion = self.mock_response_01
        answers = self.modifier.parse_response(completion)
        assert answers == ["Erster Satz.", "Zweiter Satz."]
        return

    def test_modify_dataset_gpt_modifier(self) -> None:
        # get a dataset
        data_files = ["tests/testdata/jsonl/mini/file03.jsonl"]
        dataset = datasets.load_dataset("json", data_files=data_files, split="train")
        self.modifier.max_len_prompt = 100

        # code copied from modify_dataset (adjusted: self.modifier instead of self), but without using `self.query_client`
        preds = []
        prompt = self.modifier.prompt_base
        len_prompt_base = num_tokens_from_messages(self.modifier.prompt_base)
        examples: List[str] = []
        len_examples = 0
        for record in dataset:
            current_example = self.modifier.record_to_example_line(record)
            len_current_example = num_tiktokens(
                current_example, model=self.modifier.model
            )
            len_total = len_prompt_base + len_examples + len_current_example
            print(len_total)
            if len_total > self.modifier.max_len_prompt:
                # build prompt and pass to client without current example
                prompt[-1]["content"] = "\n\n".join(examples)

                #### Fake response instead of quering API; saves money ;-)
                # response = self.query_client(prompt)
                response = self.mock_response_01

                answers = self.modifier.parse_response(response)
                # collect predictions
                preds.extend([{"pred": pred} for pred in answers])
                # reset example list and length
                examples = [current_example]
                len_examples = len_current_example
            else:
                # extend example list and length
                examples.append(current_example)
                len_examples += len_current_example

        # final query
        # build prompt and pass to client without current example
        prompt[-1]["content"] = "\n\n".join(examples)

        ### Fake response instead of quering API; saves money ;-)
        response = self.mock_response_01

        answers = self.modifier.parse_response(response)
        # collect predictions
        preds.extend([{"pred": pred} for pred in answers])

        preds_ds = datasets.Dataset.from_list(preds)
        if len(preds_ds) != len(dataset):
            print(preds)  # so that they are not lost

        else:
            ds_final = datasets.concatenate_datasets([dataset, preds_ds], axis=1)
            print(ds_final[:])

    @pytest.mark.skip
    def test_query_client(self) -> None:
        # model
        self.modifier.model = "gpt-3.5-turbo-0125"

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I give you a list and you respond with the element's index, followed by a tab, 'Hi'. List elements are separated by two newlines."},
            {"role": "user", "content": "1\tA\n\n2\tB"},
            {"role": "assistant", "content": "1\tHi\n\n2\tHi"},
            {"role": "user", "content": "1\tC\n\n2\tD\n\n3\tE"},
        ]

        response = self.modifier.query_client(prompt)
        assert response.choices[0].message.content == "1\tHi\n\n2\tHi\n\n3\tHi"

# ALT
"""

    def test__build_messages(self) -> None:
        samples = ["A", "B", "C"]
        system_prompt = "You are a helpful assistant."
        user_prompt = "Say this is a test."
        example_query = "Hey!"
        example_response = "This is a test."
        messages = self.modifier._build_prompt(
            samples,
            system_prompt,
            user_prompt,
            example_query,
            example_response,
        )
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say this is a test."},
            {"role": "user", "content": "Hey!"},
            {"role": "assistant", "content": "This is a test."},
            {"role": "user", "content": "A\n\nB\n\nC"},
        ]

    def test__build_messages_with_None(self) -> None:
        samples = ["A", "B", "C"]
        system_prompt = None
        user_prompt = None
        messages = self.modifier._build_prompt(samples, system_prompt, user_prompt)
        assert messages == [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
            {"role": "user", "content": "A\n\nB\n\nC"},
        ]

    @pytest.mark.skip
    def test_query_client(self) -> None:
        # model
        self.modifier.model = "gpt-3.5-turbo-0125"

        # query
        samples = ["1\tC", "2\tD", "3\tE"]
        self.modifier.system_prompt = "You are a helpful assistant."
        self.modifier.user_prompt = "I give you a list and you respond with the element's index, followed by a tab, 'Hi'. List elements are separated by two newlines."
        self.modifier.example_query = "1\tA\n\n2\tB"
        self.modifier.example_response = "1\tHi\n\n2\tHi"

        answers = self.modifier.query_client(samples)
        assert answers == ["1\tHi", "2\tHi", "3\tHi"]

    def test_modify_samples(self) -> None:
        samples = [
            {
                "norm": "daß es heute in der Schiffahrt.",
                "norm_tok": ["daß", "es", "heute", "in", "der", "Schiffahrt", "."],
            },
            {
                "norm": "Das nächstemal lohnt es sich allzugroß beisammenzusein.",
                "norm_tok": [
                    "Das",
                    "nächstemal",
                    "lohnt",
                    "es",
                    "sich",
                    "allzugroß",
                    "beisammenzusein",
                    ".",
                ],
            },
        ]
        samples_new = self.modifier.modify_samples(samples)
        assert samples_new == [
            {
                "norm": "daß es heute in der Schiffahrt.",
                "norm_tok": ["daß", "es", "heute", "in", "der", "Schiffahrt", "."],
                "gpt_anno": "FOO",
            },
            {
                "norm": "Das nächstemal lohnt es sich allzugroß beisammenzusein.",
                "norm_tok": [
                    "Das",
                    "nächstemal",
                    "lohnt",
                    "es",
                    "sich",
                    "allzugroß",
                    "beisammenzusein",
                    ".",
                ],
                "gpt_anno": "FOO",
            },
        ]



"""
