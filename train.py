from unittest import TestCase
from datasets import load_dataset
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from pathformer.datasets.path_dataset import (mapping_func, my_collate, causal_dataset_transform)
from pathformer.models.simple import PathTransformerModel


def build_splits(dataset):
    rez = dataset.train_test_split(test_size=0.1)
    test_pair= rez["test"].train_test_split(test_size=0.8)
    valid_set, test_set = test_pair["train"], test_pair["test"]
    return rez["train"], valid_set, test_set


dataset = load_dataset('quickdraw', "raw", split='train')
train_dataset, valid_set, test_set = build_splits(dataset)


train_prep_dataset = (train_dataset.
                       map(mapping_func, batched=True, batch_size=2, num_proc=8,
                           remove_columns=['timestamp', "key_id", "recognized", "countrycode", "word", "drawing"]))
valid_prep_dataset = (valid_set.
                       map(mapping_func, batched=True, batch_size=2, num_proc=8,
                           remove_columns=['timestamp', "key_id", "recognized", "countrycode", "word", "drawing"]))
train_dataloader = DataLoader(train_prep_dataset, batch_size=128, num_workers=4, collate_fn=my_collate, shuffle=True)
valid_dataloader = DataLoader(valid_prep_dataset, batch_size=128, num_workers=4, collate_fn=my_collate, shuffle=True)


model = PathTransformerModel()
model.training_function(train_dataloader, valid_dataloader)