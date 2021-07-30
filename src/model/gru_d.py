import torch
import torch.nn as nn 
import math
from .group_linear import CustomLinear
from .gru_cell import CustomGRUCell

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mask_size = input_size
        self.delta_size = input_size

        '''
        gamma layer for decay mechanism
        '''
        self.gamma_x_l = CustomLinear(self.delta_size, self.delta_size)
        self.gamma_h_l = CustomLinear(self.delta_size, self.delta_size)

        self.gru_cell = CustomGRUCell(self.input_size + self.mask_size, self.hidden_size)
        self.zeros = torch.zeros(self.delta_size).float().to(self.device)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 /math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, x_mask, delta, x_last_observed, x_mean, hs):

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta))).to(self.device)
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta))).to(self.device)

        x = x_mask * x + (1 - x_mask) * (delta_x * x_last_observed + (1 - delta_x) * x_mean)
        hs = delta_h * hs

        '''
        decaying hidden states and mask through hidden state compute
        '''
        inputs = torch.cat([x, x_mask], dim=-1).to(self.device)
        hs = self.gru_cell(inputs, hs)
        
        return hs



