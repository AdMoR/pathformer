import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple, Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset



class PathRNNModel(nn.Module):

    def __init__(self, n_command: int = 3, command_embedding_size: int = 4,  d_coords: int = 2,  d_hid: int = 512,
                 nlayers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'PathRNN'
        # X, C => I
        self.embedding = torch.nn.Embedding(n_command, command_embedding_size)
        self.linear_pre_hidden = torch.nn.Linear(command_embedding_size + d_coords, d_hid)
        # H + I => H_t+1
        self.linear_hidden = torch.nn.Linear(d_hid, d_hid)
        # Get next coords
        self.linear_coords = torch.nn.Linear(d_hid, d_coords)
        self.linear_command = torch.nn.Linear(d_hid, n_command)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for linear in [self.linear_pre_hidden, self.linear_hidden, self.linear_command, self.linear_coords]:
            linear.bias.data.zero_()
            linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Dict[str, Tensor], hidden: torch.tensor = None,
                src_mask: Tensor = None) -> Dict[str, Tensor]:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # TODO
        cmd_input = torch.LongTensor(src["command"])
        print("ok")
        coord_input = torch.Tensor(src["coord"])
        B, H, _ = cmd_input.shape
        src_cmd = self.embedding(cmd_input).squeeze(-2)
        src = torch.cat([src_cmd, coord_input], axis=-1)
        pre_hidden = self.linear_pre_hidden(src)

        if hidden is None:
            hidden = torch.zeros_like(pre_hidden)

        new_hidden = self.linear_hidden(pre_hidden + hidden)

        output_command = self.linear_command(new_hidden)
        output_coords = torch.softmax(self.linear_coords(new_hidden), axis=-1)
        return {"command": output_command, "coord": output_coords}

