import csv
import pytest
import unittest

import datasets

from transnormer_data.modifier.gpt_modifier import GPTModifier


class GPTModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = GPTModifier("","","","","") # dummy vars

    def tearDown(self) -> None:
        pass

    def test__build_messages(self) -> None:
        samples = ["A", "B", "C"]
        system_prompt = "You are a helpful assistant."
        user_prompt = "Say this is a test."
        example_query = "Hey!"
        example_response = "This is a test."
        messages = self.modifier._build_messages(
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
        messages = self.modifier._build_messages(samples, system_prompt, user_prompt)
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
        assert(answers == ["1\tHi", "2\tHi", "3\tHi"])



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
            ]
            }
        ]
        samples_new = self.modifier.modify_samples(samples)
        assert samples_new == [
            {
            "norm": "daß es heute in der Schiffahrt.",
            "norm_tok": ["daß", "es", "heute", "in", "der", "Schiffahrt", "."],
            "gpt_anno": "FOO"
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
            "gpt_anno": "FOO"
            }
        ]

    def test_modify_dataset(self) -> None:
        pass



    # self.model = "gpt-3.5-turbo-0125"
    # self.system_prompt = "You are an expert for historical language."
    # self.user_prompt = "Say: This is a test."
    # self.example_query, self.example_response = (
    #     None,
    #     None,
    # )
    # Look like this:
    # {"role": "user", "content": "Who won the world series in 2020?"},
    # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
