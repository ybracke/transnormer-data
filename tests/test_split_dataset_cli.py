import os
import tempfile
import json
import shutil
import unittest
from collections import Counter
from transnormer_data.cli.split_dataset import (
    load_document_metadata,
    get_decade,
    group_documents_by_decade_genre,
    split_authors,
    create_splits,
    main,
)


class SplitDatasetTester(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        # Create temporary test JSONL files
        filepaths = []
        for i in range(10):
            filepath = os.path.join(self.temp_dir, f"test_{i}.jsonl")
            with open(filepath, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "basename": f"test_{i}",
                            "date": 1700 + i,
                            "genre": "Fiction::Historical",
                            "author": f"Author_{i}",
                        }
                    )
                    + "\n"
                )
            filepaths.append(filepath)
        return filepaths

    def test_load_document_metadata(self):
        filepaths = self.create_test_files()
        documents_meta = load_document_metadata(self.temp_dir, lambda x: True)
        self.assertEqual(len(documents_meta), 10)

    def test_get_decade(self):
        self.assertEqual(get_decade(1705), 1700)
        self.assertEqual(get_decade(1713), 1710)
        self.assertEqual(get_decade(1799), 1790)

    def test_group_documents_by_decade_genre(self):
        filepaths = self.create_test_files()
        documents_meta = load_document_metadata(self.temp_dir, lambda x: True)
        groups = group_documents_by_decade_genre(documents_meta)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[(1700, "Fiction")]), 10)

    def test_split_authors(self):
        authors = [
            "Author_1",
            "Author_2",
            "Author_3",
            "Author_4",
            "Author_5",
            "Author_6",
            "Author_7",
            "Author_8",
            "Author_9",
            "Author_10",
        ]
        num_works_per_author = Counter(authors)
        train_authors, val_authors, test_authors = split_authors(
            authors, num_works_per_author
        )
        self.assertEqual(len(train_authors), 8)
        self.assertEqual(len(val_authors), 1)
        self.assertEqual(len(test_authors), 1)

    def test_create_splits(self):
        filepaths = self.create_test_files()
        documents_meta = load_document_metadata(self.temp_dir, lambda x: True)
        groups = group_documents_by_decade_genre(documents_meta)
        train_docs, val_docs, test_docs = create_splits(groups)
        self.assertEqual(len(train_docs), 8)
        self.assertEqual(len(val_docs), 1)
        self.assertEqual(len(test_docs), 1)
