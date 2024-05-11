from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from pathformer.datasets.path_dataset import (mapping_func, my_collate, causal_dataset_transform)
from pathformer.models.simple import PathTransformerModel
from torch.utils.tensorboard import SummaryWriter


def build_splits(dataset):
    rez = dataset.train_test_split(test_size=0.1)
    test_pair= rez["test"].train_test_split(test_size=0.8)
    valid_set, test_set = test_pair["train"], test_pair["test"]
    return rez["train"], valid_set, test_set


dataset = load_dataset('quickdraw', "raw", split='train')
train_dataset, valid_set, test_set = build_splits(dataset)


train_prep_dataset = (train_dataset.select(range(1000000))
                       .with_transform(mapping_func, columns=["drawing", "word"]))
valid_prep_dataset = (valid_set
                       .with_transform(mapping_func, columns=["drawing", "word"]))
train_dataloader = dataloader = DataLoader(train_prep_dataset, batch_size=128, num_workers=6, collate_fn=my_collate,
                                           pin_memory=True)
valid_dataloader = DataLoader(valid_prep_dataset, batch_size=64, num_workers=2, collate_fn=my_collate, pin_memory=True)
writer = SummaryWriter()

with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    model = PathTransformerModel(writer=writer).cuda()
    model.training_function(train_dataloader, valid_dataloader)