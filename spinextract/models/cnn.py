"""Model wrappers.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Dict, List
from itertools import chain

import torch
from torch import nn as nn

import timm


class MLP(nn.Module):
    """MLP for classification head.

    Forward pass returns a tensor.
    """

    def __init__(self, n_in: int, hidden_layers: List[int],
                 n_out: int) -> None:
        super().__init__()
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]

        layers_list = list(
            chain.from_iterable((nn.Linear(a, b), nn.ReLU())
                                for a, b in zip(layers_in, layers_out)))[:-1]
        self.layers = nn.Sequential(*layers_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, n_in]
        Returns:
            output: Tensor, shape [batch_size, n_in]
        """
        batch_size = src.size(0)
        
        src = src.view(batch_size, -1, self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feedforward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = src.transpose(0, 1)
        src = src.reshape(batch_size, -1)
        
        return src

class TransformerHead(nn.Module):
    def __init__(self, n_in: int, hidden_layers: List[int], n_out: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.n_in = n_in
        self.hidden_layers = hidden_layers
        
        # Calculate appropriate d_model (should divide n_in evenly)
        potential_d_models = [32] # could be any, but 32 is a good start
        self.d_model = None
        for d in potential_d_models:
            if n_in % d == 0:
                self.d_model = d
                break
        if self.d_model is None:
            raise ValueError(f"Could not find suitable d_model that divides {n_in}")
        
        self.seq_length = n_in // self.d_model
        
        # Initial projection
        self.input_projection = nn.Linear(n_in, n_in)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.d_model, num_heads, self.d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # FFN layers
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]
        ffn_layers = []
        for a, b in zip(layers_in, layers_out):
            ffn_layers.extend([nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)])
        ffn_layers = ffn_layers[:-2]
        self.ffn = nn.Sequential(*ffn_layers)
        
        self.output = nn.Linear(layers_out[-1], n_out)
        self.norm = nn.LayerNorm(n_out)
        
        self.dropout = nn.Dropout(dropout)
        self.num_out = n_out
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if isinstance(p, nn.Linear):
                if 'output' in name:
                    # Final classification layer
                    nn.init.normal_(p.weight, mean=0, std=0.02)
                elif any(f'ffn.{i}' in name for i in range(0, len(self.hidden_layers) * 3, 3)):
                    # Hidden layers in classification MLP
                    nn.init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # Transformer layers
                    nn.init.xavier_uniform_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)
            elif isinstance(p, nn.LayerNorm):
                nn.init.ones_(p.weight)
                nn.init.zeros_(p.bias)
            elif isinstance(p, nn.MultiheadAttention):
                for param in [p.in_proj_weight, p.out_proj.weight]:
                    nn.init.xavier_uniform_(param)
                for param in [p.in_proj_bias, p.out_proj.bias]:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_in)
        Returns:
            output: Tensor of shape (batch_size, n_out)
        """
        x = self.input_projection(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Classification head
        x = self.ffn(x)
        x = self.output(x)
        x = self.norm(x)
        
        if self.num_out == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)
        
        return x  


class Classifier(nn.Module):
    """A network consists of a backbone and a classification head.

    Forward pass returns a dictionary, which contains both the logits and the
    embeddings.
    """

    def __init__(self, backbone: callable, head: callable) -> None:
        """Initializes a Classifier.
        Args:
            backbone: the backbone, either a class, a function, or a parital.
                It defaults to resnet50.
            head: classification head to be attached to the backbone.
        """
        super().__init__()
        self.bb = backbone()
        self.head = head()

    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass"""
        bb_out = self.bb(x)
        return {'logits': self.head(bb_out), 'embeddings': bb_out}


def vit_backbone(params):
    """Function used to call ViT model from PyTorch Image Models.

    ViT source code in PyTorch Image Models (timm):
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py.
    """
    model = timm.create_model(**params)
    model.head = nn.Identity()
    model.num_out = model.embed_dim
    return model


class Network(nn.Module):

    def __init__(self, backbone: callable, dim=384, pred_dim=96):
        super(Network, self).__init__()
        self.bb = backbone()

        # projection layer same as original paper
        prev_dim = self.bb.num_out
        self.proj = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, dim, bias=True),
            nn.BatchNorm1d(dim, affine=False))  # output layer
        self.proj[
            6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.pred = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass, Takes in a single augmented image and runs a single branch of network"""
        bb_out = self.proj(self.bb(x, **kwargs))
        pred_out = self.pred(bb_out)

        return {'proj': bb_out, 'pred': pred_out}


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new