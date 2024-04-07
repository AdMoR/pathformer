from unittest import TestCase
from datasets import load_dataset
import numpy as np
import json
import os

from pathformer.datasets.path_dataset import (mapping_func)

from pathformer.models.simple import PathRNNModel

class TestModel(TestCase):

    def test_model_forward(self):
        dataset = load_dataset('quickdraw', "raw", split='train')
        proc_dataset = (dataset.select(range(10000)).
                        map(mapping_func, batched=True, num_proc=8, remove_columns=['timestamp', "key_id"]))
        model = PathRNNModel()
        sample = proc_dataset[0:2]
        rez = model(sample)