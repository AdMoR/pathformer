import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple, Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset



class PathTransformerModel(nn.Module):

    def __init__(self, n_command: int = 4, command_embedding_size: int = 4,  d_coords: int = 2,
                 nlayers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'PathTransformerModel'
        d_model = command_embedding_size + d_coords
        self.embedding = torch.nn.Embedding(n_command, command_embedding_size, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(d_model / 2))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.linear_coords = torch.nn.Linear(d_model, d_coords)
        # n_command includes the padding token, so should we include it in the softmax => porbably not
        # But then how can it be handled seamlessly ??
        self.linear_command = torch.nn.Linear(d_model, n_command)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for linear in [self.linear_command, self.linear_coords]:
            linear.bias.data.zero_()
            linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Expectations:
        src : command [N x seq x 1]
              coord [N x seq x 2]
              command_target [N x 1]
              coord_target [N x 2]
        """
        cmd_input = src["command"].long()
        coord_input = src["coord"]
        src_cmd = self.embedding(cmd_input).squeeze(-2)
        src = torch.cat([src_cmd, coord_input], dim=-1)
        out = self.encoder(src)
        pred_out = torch.mean(out, dim=1)
        output_coords = self.linear_coords(pred_out)
        output_command = torch.softmax(self.linear_command(pred_out), dim=-1)
        return {"command": output_command, "coord": output_coords}

    def iter_loop(self, src: Dict[str, Tensor], lambda_: float = 1.0):
        out = self.forward(src)

        target_coord = src["coord_target"]
        target_cmd = src["command_target"]

        # L2 norm on coord
        loss_1 = torch.linalg.norm(out["coord"] - target_coord, 2)
        # Log likelihood on the command head
        loss_2 = -torch.mean(torch.log(torch.gather(out["command"], 1, target_cmd.long())))

        loss = loss_1 + lambda_ * loss_2
        return loss