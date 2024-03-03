from typing import List, Dict
import numpy as np
import ndjson
import dataclasses
import enum


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

    def to_tensor(self):
        return {"command": self.command_name.value, "coord": self.coords}

    def render(self):
        pass


def build_positions_from_ndjson(drawing_blob: Dict) -> List[np.array]:
    out = list()
    for stroke in drawing_blob:
        coords = list(zip(*stroke[:2]))
        out.append(np.array(coords))
    return out


def build_stroke_transition(stroke_array: List[List[np.array]]) -> List[CoordCommand]:
    """
    in : stroke: List[(index, x, y)]
    out : stroke List[(command(M if i==0 else L), dx, dy
    """
    commands = list()
    for stroke in stroke_array:
        for i, path in enumerate(stroke):
            if len(path) != 2:
                raise Exception("Invalid coordinate shape")
            if i == 0:
                commands.append(CoordCommand(LineCommand.M, path))
            else:
                diff = stroke[i] - stroke[i - 1]
                commands.append(CoordCommand(LineCommand.L, diff))
    commands.append(CoordCommand(LineCommand.Z, np.array([0, 0])))
    return commands


def build_dataset(path):
    data = ndjson.load(open(path))
    documents = list()
    for i in range(len(data)):
        array_of_strokes = build_positions_from_ndjson(data[i]["drawing"])
        array_of_commands = build_stroke_transition(array_of_strokes)
        documents.append(array_of_commands)
    return documents


