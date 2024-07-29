import unittest

from transnormer_data.modifier.replace_raw_modifier import (
    ReplaceRawModifier,
)


class ReplaceRawModifierTester(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = None
        self.modifier = ReplaceRawModifier(mapping_files=[])
        # self.mapping_files = ["tests/testdata/type-replacements/old2new.tsv"]

    def tearDown(self) -> None:
        pass

    def test_load_mapping_single_file(self) -> None:
        mapping_files = ["tests/testdata/raw-replacement/corrected-sents.csv"]
        mapping = self.modifier._load_corrected_samples(
            mapping_files, ["basename", "par_idx"], "norm_correct"
        )
        print(mapping)

    # def test_load_mapping_two_files(self) -> None:
    #     target_mapping = {
    #         "daß": "dass",
    #         "muß": "muss",
    #         "Schiffahrt": "Schifffahrt",
    #         "sehn": "sehen",
    #         "glühn": "glühen",
    #     }
    #     mapping_files = [
    #         "tests/testdata/type-replacements/old2new.tsv",
    #         "tests/testdata/type-replacements/error2correct.tsv",
    #     ]
    #     mapping = self.modifier._load_replacement_mapping(mapping_files)
    #     assert target_mapping == mapping

    # def test_load_mapping_broken_file(self) -> None:
    #     mapping_files = ["tests/testdata/type-replacements/broken.tsv"]
    #     with pytest.raises(csv.Error):
    #         _ = self.modifier._load_replacement_mapping(mapping_files)
