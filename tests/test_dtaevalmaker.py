import os
from typing import List
import unittest

from lxml import etree
import datasets

from transnormer_data.maker.dta_eval_maker import DtaEvalMaker

TMPDIR = "tests/testdata/tmp"


class DtaEvalMakerTester(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir_data = "tests/testdata/dtaeval/xml"
        self.input_dir_meta = "tests/testdata/metadata/metadata_dtak.jsonl"
        self.output_dir = TMPDIR
        self.target_properties = {
            "basename": str,
            "par_idx": str,
            "date": int,
            "genre": str,
            "author": str,
            "title": str,
            "is_bad": bool,
            "orig": str,
            "orig_tok": List[str],
            "orig_class": List[str],
            "orig_ws": List[bool],
            "orig_spans": List[List[int]],
            "norm": str,
            "norm_tok": List[str],
            "norm_ws": List[bool],
            "norm_spans": List[List[int]],
            "alignment": List[List[int]],
        }

        self.maker = DtaEvalMaker(
            self.input_dir_data, self.input_dir_meta, self.output_dir
        )
        self.dataset: datasets.Dataset = self.maker.make(save=True)

    def tearDown(self) -> None:
        # Remove files that were created during training
        for root, dirs, files in os.walk(TMPDIR, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            else:
                os.rmdir(root)

    def test_custom_split(self) -> None:
        """Test the custom split function used by DtaEvalMaker"""
        result = self.maker.custom_split("Haus")
        assert result == (["Haus"], False)
        result = self.maker.custom_split("bin_es")
        assert result == (["bin", "es"], True)
        result = self.maker.custom_split("bin_es nicht")
        assert result == (["bin", "es", "nicht"], True)

    def test_file_creation(self) -> None:
        """Test whether correctly named files have been created by the Maker"""
        path_01 = f"{TMPDIR}/brentano_kasperl_1838.jsonl"
        path_02 = f"{TMPDIR}/fontane_stechlin_1899.jsonl"
        assert os.path.isfile(path_01)
        assert os.path.isfile(path_02)

    def test_data_structure(self) -> None:
        """Test whether the dataset's features are identical to the target features"""
        assert self.dataset.features.keys() == self.target_properties.keys()

    def test_shape(self) -> None:
        """Test whether dataset's shape is correct"""
        assert self.dataset.shape == (28, len(self.target_properties))

    def test_alignments(self) -> None:
        """Check whether the correct alignments were computed"""
        # Target alignments for brentano
        target_alignments = [
            [[i, i] for i in range(75)],
            [[i, i] for i in range(9)]
            + [[9, 9], [10, 9]]
            + [[i, i - 1] for i in range(11, 17)],
            [[i, i] for i in range(33)],
            [[i, i] for i in range(49)],
            [[i, i] for i in range(35)],
            [[i, i] for i in range(6)],
            [[i, i] for i in range(13)],
            [[i, i] for i in range(32)] + [[32, 32], [33, 32], [34, 33]],
            [[i, i] for i in range(23)],
            [[i, i] for i in range(13)],
            [[i, i] for i in range(13)],
            [[i, i] for i in range(36)],
            [[i, i] for i in range(8)],
            [[i, i] for i in range(20)],
            [[i, i] for i in range(32)],
            [[i, i] for i in range(18)],
            [[i, i] for i in range(14)],
            [[0, 0], [1, 1], [2, 2], [3, 2], [4, 2]]
            + [[i, i - 2] for i in range(5, 20)],
        ]
        for example, target_alignment in zip(
            self.dataset.filter(
                lambda example: example["basename"].startswith("brentano_kasperl_1838")
            ),
            target_alignments,
        ):
            assert example["alignment"] == target_alignment

    def test_create_example_from_s_with_join(self) -> None:
        """Test example creation from a single sentence"""
        tree = etree.parse(
            os.path.join(self.input_dir_data, "brentano_kasperl_1838.xml")
        )
        s = tree.findall(".//s")[-1]  # last sentence in brentano
        target = (
            [
                "ich",
                "bin",
                "acht",
                "und",
                "achtzig",
                "Jahr",
                "alt",
                ",",
                "und",
                "der",
                "Herzog",
                "wird",
                "mich",
                "gewiß",
                "nicht",
                "von",
                "ſeiner",
                "Schwelle",
                "treiben",
                ".",
            ],
            [
                "Ich",
                "bin",
                "achtundachtzig",
                "Jahr",
                "alt",
                ",",
                "und",
                "der",
                "Herzog",
                "wird",
                "mich",
                "gewiß",
                "nicht",
                "von",
                "seiner",
                "Schwelle",
                "treiben",
                ".",
            ],
            [
                "LEX",
                "LEX",
                "JOIN",
                "JOIN",
                "JOIN",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
                "LEX",
            ],
        )
        result = self.maker._create_example_from_s(s)
        assert result == target

    def test_create_example_from_s_with_hypen_join(self) -> None:
        """Test example creation from a single sentence"""

        xml_string = """
<s>
<w class="LEX" new="In" old="In" pok="1"/>
<w class="LEX" new="der" old="der" pok="1"/>
<w class="LEX" new="Allgemeinen" old="allgemeinen" pok="1"/>
<w class="LEX" new="Zeitung" old="Zeitung" pok="1"/>
<w class="LEX" new="stehen" old="ſtehen" pok="1"/>
<w class="LEX" new="Berichte" old="Berichte" pok="1"/>
<w class="LEX" new="über" old="über" pok="1"/>
<w class="LEX" new="die" old="die" pok="1"/>
<w bad="1" class="JOIN" new="Ständeversammlungen" old="Stände- Verſammlungen" pok="1" seen="1">
    <w bad="1" class="JOIN" new="Stände" old="Stände-" pok="1" seen="1"/>
    <w class="JOIN" new="Versammlungen" old="Verſammlungen" pok="1" seen="1"/>
</w>
<w class="LEX" new="." old="." pok="1"/>
</s>"""
        target = (
            [
                "In",
                "der",
                "allgemeinen",
                "Zeitung",
                "ſtehen",
                "Berichte",
                "über",
                "die",
                "Stände-Verſammlungen",
                ".",
            ],
            [
                "In",
                "der",
                "Allgemeinen",
                "Zeitung",
                "stehen",
                "Berichte",
                "über",
                "die",
                "Ständeversammlungen",
                ".",
            ],
            ["LEX", "LEX", "LEX", "LEX", "LEX", "LEX", "LEX", "LEX", "LEX", "LEX"],
        )
        s = etree.fromstring(xml_string)
        result = self.maker._create_example_from_s(s)
        assert result == target

    def test_join_wrongly_splitted_tokens(self) -> None:
        """Test token joining function"""
        tokens = ["Versammlungs-", "Freiheits-", "Gesetz"]
        target = ["Versammlungs-Freiheits-Gesetz"]
        result = self.maker.join_wrongly_splitted_tokens(tokens)
        assert result == target
