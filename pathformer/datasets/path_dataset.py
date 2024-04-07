from typing import List, Dict
import os
import json
import numpy as np
import ndjson
import dataclasses
import enum
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch


class LineCommand(enum.Enum):
    """
        M = "MoveTo"
        L = "LineTo"
        Z = "End"
    """
    M = 0
    L = 1
    Z = 2


@dataclasses.dataclass
class CoordCommand:
    command_name: LineCommand
    coords: np.array

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        return {"command": torch.tensor(self.command_name.value).unsqueeze(0),
                "coord": torch.tensor(self.coords).unsqueeze(0)}


@dataclasses.dataclass
class CoordDrawing:
    strokes: List[CoordCommand]
    category: str = None

    def to_batch_tensor(self):
        commands = torch.cat([self.strokes[i].to_tensor()["command"]
                              for i in range(len(self.strokes))], axis=0).unsqueeze(-1)
        coords = torch.cat([self.strokes[i].to_tensor()["coord"] for i in range(len(self.strokes))], axis=0)
        return {"command": commands, "coord": coords}

    def render(self):
        pass


def build_positions_from_ndjson(drawing_blob: Dict) -> List[np.array]:
    out = list()
    for stroke in drawing_blob:
        coords = list(zip(*stroke[:2]))
        out.append(np.array(coords))
    return out

def build_position_from_hf_elem(drawing_blob: Dict) -> List[np.array]:
    """
    {
      "x": [
          [1, 3, 5, 88], [33, 4, 90]
      ],
      "y": ....
    }
    """
    out = list()
    for array_x, array_y in zip(drawing_blob["x"], drawing_blob["y"]):
        coords = list(zip(array_x, array_y))
        out.append(np.array(coords))
    return out

def build_stroke_transition(stroke_array: np.array, category: str) -> CoordDrawing:
    """
    in : stroke: List[(index, x, y)]
    out : stroke List[(command(M if i==0 else L), x, y
    """
    commands = list()
    for stroke in stroke_array:
        for i, path in enumerate(stroke):
            if i == 0:
                commands.append(CoordCommand(LineCommand.M, path))
            else:
                diff = stroke[i] - stroke[i - 1]
                commands.append(CoordCommand(LineCommand.L, diff))
    commands.append(CoordCommand(LineCommand.Z, np.array([0, 0])))
    return CoordDrawing(commands)


def build_dataset(path):
    data = ndjson.load(open(path))
    documents = list()
    for i in range(len(data)):
        array_of_strokes = build_positions_from_ndjson(data[i]["drawing"])
        array_of_commands = build_stroke_transition(array_of_strokes)
        documents.append(array_of_commands)
    return documents


def find_file_i_for_sample_index(cumsum_index: List[int], dataset_index: int):
    """
    How does it work :
    we want sample index from the dataset, the cumsum tells us in which file it is located
    ex:
    index = 733
    cumsum_inex = [133, 546, 833]

    thus the right file_index is 2
    """
    i = 0
    while dataset_index >= cumsum_index[i] and i < len(cumsum_index) - 1:
        i += 1
    if i == 0:
        reminder = dataset_index
    else:
        reminder = dataset_index - (cumsum_index[i - 1])
    return i, reminder


def mapping_func(dict_object: Dict):
    """
    Used in batch mode, input received is as following :
    "drawing": [{"t": [[1, 4, 5], [88, 92]], "x": [[23, 45, 23], [...]]}]
    """
    results = list()
    for drawing, category in zip(dict_object["drawing"], dict_object["word"]):
        results.append(build_stroke_transition(build_position_from_hf_elem(drawing), category).to_batch_tensor())
    keys = list(results[0].keys())
    final_results = {k: [elem[k] for elem in results] for k in keys}
    return final_results


def my_collate(batch, default_value_dict=None):
    if default_value_dict is None:
        default_value_dict = defaultdict(lambda: -1)
    batch_size = len(batch)
    max_length = max(np.array(e["coord"]).shape[0] for e in batch)
    out_dict = dict()
    for k in batch[0].keys():
        out_dict[k] = torch.Tensor(
            [np.pad(e[k], pad_width=((0, max_length - len(e[k])), (0, 0)), mode="constant",
                    constant_values=default_value_dict[k])
             for e in batch])
    return out_dict


class DrawingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:

        """
        # Find all the files in the folder
        self.all_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith("ndjson")]
        # Count the total number of sample in the whole dataset
        # The cumsum index allows to know in which file each sample will be
        memory_file = f"{root_dir}/cumsum_file.json"
        if not os.path.exists(memory_file):
            self.cumsum_index = np.cumsum([len(ndjson.load(open(f))) for f in self.all_files]).tolist()
            json.dump(self.cumsum_index, open(memory_file, "w"))
        else:
            self.cumsum_index = json.load(open(memory_file, "r"))


    def __len__(self):
        return self.cumsum_index[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        array_to_open = set(map(lambda i: find_file_i_for_sample_index(self.cumsum_index, i), idx))
        data_array = {file_index: ndjson.load(open(self.all_files[file_index])) for (file_index, _) in array_to_open}

        samples = list()
        for (file_index, reminder) in array_to_open:
            out = build_positions_from_ndjson(data_array[file_index][reminder]["drawing"])
            rez = build_stroke_transition(out, data_array[file_index][reminder]["word"])
            samples.append(rez.to_batch_tensor())

        return samples