from numpy.core.fromnumeric import std
import torch
import torch.nn as nn 
import math
from .group_linear import GroupLinear

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mask_size = input_size
        self.delta_size = input_size

        self.input_hidden = GroupLinear(input_size + hidden_size+ self.mask_size, hidden_size *3, num_blocks)
        self.hidden_hidden = GroupLinear(hidden_size, hidden_size *3, num_blocks)
        self.bias = nn.Parameter(torch.Tensor(hidden_size *3))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 /math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self):
        pass