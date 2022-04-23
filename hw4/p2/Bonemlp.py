import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

class MLP(nn.Module):
    def __init__(self, back, fin_size, hidden_size = 1000):
        super().__init__()
        self.back = back
        self.net = nn.Sequential(
            # nn.Linear(hidden_size, 2000),
            # nn.BatchNorm1d(2000),
            # nn.ReLU(),
            nn.Linear(hidden_size, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, fin_size)     
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.r1 = nn.ReLU()

    def forward(self, x):
        x = self.back(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.net(x)
        return x