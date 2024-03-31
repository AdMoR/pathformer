import math
from typing import Tuple, Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PathTransformerModel(nn.Module):

    def __init__(self, n_command: int = 3, d_model: int = 8, d_coords=2, nhead: int = 8, d_hid: int = 512,
                 nlayers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'PathTransformer'
        # Only for the coords part ?
        self.pos_encoder = PositionalEncoding(d_coords, dropout)
        # Only for the
        self.embedding = nn.Embedding(n_command, d_model - d_coords)
        self.d_model = d_model

        # Transformer part
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Outputs :
        self.linear_command = nn.Linear(d_model, n_command)
        self.linear_coords = nn.Linear(d_model, d_coords)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for linear in [self.linear_command, self.linear_coords]:
            linear.bias.data.zero_()
            linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Dict[str, Tensor], src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src_cmd = self.embedding(src["command"]) * math.sqrt(self.d_model)
        src_coord = self.pos_encoder(src["coord"].shape[0])
        print(src_cmd.shape, src_coord.shape)
        src = torch.cat([src_cmd, src_coord], axis=-1)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output_command = self.linear_command(output)
        output_coords = self.linear_command(output)
        return {"command": output_command, "coord": output_coords}

