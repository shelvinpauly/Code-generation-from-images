import os
import unittest
from pathlib import Path
from transformers import AutoTokenizer, ViTFeatureExtractor
from dataloader import (
    split_paths_train_val_test,
)


class TestDatasetFunctions(unittest.TestCase):
    def test_split_paths_train_val_test(self):
        sample_paths = [i for i in range(100)]
        train, val, test = split_paths_train_val_test(sample_paths)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
