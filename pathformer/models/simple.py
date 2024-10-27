import math
import os
import time
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Any
from unittest import mock

import torch
from torch import nn, Tensor

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = mock.Mock


class PathTransformerModel(nn.Module):

    def __init__(self, n_command: int = 4, command_embedding_size: int = 4,  d_coords: int = 2,
                 nlayers: int = 1, dropout: float = 0.5, writer: SummaryWriter = None):
        super().__init__()
        self.model_type = 'PathTransformerModel'
        self.hyper_params = {"n_commands": n_command, "embedding_size": command_embedding_size, "d_coord": d_coords, "n_layers":
nlayers}
        d_model = command_embedding_size + d_coords
        self.embedding = torch.nn.Embedding(n_command, command_embedding_size, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(d_model / 2), dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.linear_coords = torch.nn.Linear(d_model, d_coords)
        # n_command includes the padding token, so should we include it in the softmax => porbably not
        # But then how can it be handled seamlessly ??
        self.linear_command = torch.nn.Linear(d_model, n_command)
        self.init_weights()
        self.writer = writer if writer is not None else mock.Mock()

    @property
    def name(self):
        return "_".join([self.model_type, *[f"{k}={v}" for k, v in self.hyper_params.items()]])

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
        cmd_input = src["command"]
        coord_input = src["coord"]
        src_cmd = self.embedding(cmd_input).squeeze(-2)
        src = torch.cat([src_cmd, coord_input], dim=-1)
        out = self.encoder(src)
        pred_out = torch.mean(out, dim=1)
        output_coords = self.linear_coords(pred_out)
        output_command = torch.softmax(self.linear_command(pred_out), dim=-1)
        return {"command": output_command, "coord": output_coords}

    def iter_loop(self, src: Dict[str, Tensor], lambda_: float = 1.0, writer_log=False, step: int = None):
        if writer_log:
            tt = time.time()
        out = self.forward(src)

        target_coord = src["coord_target"]
        target_cmd = src["command_target"]

        # L2 norm on coord
        loss_1 = torch.sum((out["coord"] - target_coord) ** 2)
        # Log likelihood on the command head
        loss_2 = -torch.mean(torch.log(torch.gather(out["command"], 1, target_cmd)))

        loss = loss_1 + lambda_ * loss_2

        if writer_log:
            self.writer.add_scalars("train/loss",
                                    {"classif_loss": loss_2, "regression_loss": loss_1,
                                     "combined_loss": loss}, step)
            self.writer.add_scalar("Time/train/batch", time.time() - tt, step)
            for k, v in self.named_parameters():
                self.writer.add_histogram(f"model/weights/{k}", v.data.detach().cpu().numpy())

        return loss

    def training_function(self, train_dataloader: torch.utils.data.DataLoader,
                          valid_dataloader: torch.utils.data.DataLoader, n_epochs: int = 1, lr: float = 0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        hparams = self.hyper_params
        hparams.update({"lr": lr})
        self.writer.add_hparams(hparams, {'hparam/accuracy': 0, 'hparam/loss': 0})

        for i in range(n_epochs):
            print(f"Epoch {i}")
            for j, batch in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                for k in batch.keys():
                    batch[k] = batch[k].cuda()
                loss = self.iter_loop(batch, writer_log=j % 100 == 0, step=i * len(train_dataloader) + j)
                loss.backward()
                optimizer.step()

                if j % max(2, int(len(train_dataloader) / 10)) == 0:
                    with torch.no_grad():
                        losses = list()
                        for k, valid_batch in enumerate(valid_dataloader):
                            for k in valid_batch.keys():
                                valid_batch[k] = valid_batch[k].cuda()
                            losses.append(self.iter_loop(valid_batch, writer_log=False).detach().cpu().numpy())
                        self.writer.add_scalar("valid_loss", np.mean(losses))
                        self.save_model()


    def save_model(self, model_dir="./model_directory"):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(self.state_dict(), f"./{model_dir}/{self.name}.pt")