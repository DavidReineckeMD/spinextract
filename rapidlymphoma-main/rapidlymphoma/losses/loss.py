import torch
from torch import nn



class SimilarityLoss(nn.Module):
    """ Computes BYOL's simple similarity loss"""

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def criterion(self, x, y):
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
