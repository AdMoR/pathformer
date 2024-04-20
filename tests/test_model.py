from unittest import TestCase
from datasets import load_dataset
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from pathformer.datasets.path_dataset import (mapping_func, my_collate, causal_dataset_transform)
from pathformer.models.simple import PathTransformerModel

class TestModel(TestCase):

    def test_model_forward(self):
        dataset = load_dataset('quickdraw', "raw", split='train')
        sub_dataset = (dataset.select(range(10000)).
                       map(mapping_func, batched=True, batch_size=2, num_proc=8,
                           remove_columns=['timestamp', "key_id", "recognized", "countrycode", "word", "drawing"]))
        dataloader = DataLoader(sub_dataset, batch_size=32, num_workers=1, collate_fn=my_collate, shuffle=True)
        batch = next(iter(dataloader))
        model = PathTransformerModel()
        rez = model(batch)

    def test_model_forward(self):
        dataset = load_dataset('quickdraw', "raw", split='train')
        sub_dataset = (dataset.select(range(10000)).
                       map(mapping_func, batched=True, batch_size=2, num_proc=8,
                           remove_columns=['timestamp', "key_id", "recognized", "countrycode", "word", "drawing"]))
        dataloader = DataLoader(sub_dataset, batch_size=32, num_workers=1, collate_fn=my_collate, shuffle=True)
        batch = next(iter(dataloader))
        model = PathTransformerModel()
        loss = model.iter_loop(batch)
        print(loss)