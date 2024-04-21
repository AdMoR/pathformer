from typing import Dict
import torch

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src):
    """
    Expectation for the model
    model: Func[src -> {"coord": Tensor(N, 2), "command": Tensor(N, 1)}]
    """
    pass


def recombine(a, b):
    pass


def draw_sequence(sequence: Dict[str, torch.Tensor]):
    """
    Expected format of sequence : {"coord": Tensor(N, S, 2), "command": Tensor(N, S, 1)}
    """
    N, _ = sequence["coord"].shape
    for i in range(N):
        xy_seq = sequence["coord"][i, :, :].numpy()
        command_seq = sequence["coord"][i, :, 0].numpy()
        array_of_strokes = recombine(xy_seq, command_seq)

