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