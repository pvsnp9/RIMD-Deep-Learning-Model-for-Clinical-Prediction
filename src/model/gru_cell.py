import torch
import torch.nn as nn
import math

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.Tensor(input_size,hidden_size * 3))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 3 ))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden_state):
        # X [batch_size, time_stamp, features] time_stamp = 1 squeezed
        x = x.squeeze()
        h_t = hidden_state

        input_gates = x @ self.W +  self.bias
        hidden_gates = h_t @ self.U + self.bias

        r_t = torch.sigmoid(input_gates[:, :self.hidden_size] + hidden_gates[:, :self.hidden_size]) # reset gate 
        z_t = torch.sigmoid(input_gates[:, self.hidden_size: self.hidden_size * 2] + hidden_gates[:, self.hidden_size: self.hidden_size * 2]) # update gate 
        c_t = torch.tanh(input_gates[:, self.hidden_size * 2: ] + (r_t * hidden_gates[:, self.hidden_size *2:]))  #candidate gate
        
        h_t = c_t + z_t * (h_t - c_t)
        return h_t



# if __name__ == '__main__':
#     gru = CustomGRUCell(12, 50)
#     h_t =torch.zeros(2, 50)
#     x = torch.randn(2,1,12)
#     h_s= gru(x, h_t)
#     print(f'{h_s.size()} \n {h_s}')
    # print(f'hs: {h_s.size()}, cs: {c_s.size()}')