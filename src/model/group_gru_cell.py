import torch 
import torch.nn as nn
import math
from .group_linear import GroupLinear

class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinear(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinear(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
		input: x (batch_size, num_grus, input_size)
			   hidden (batch_size, num_grus, hidden_size)
		output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy



# if __name__ == '__main__':
#     gru = GroupGRUCell(12, 50, 3)
#     h_t =torch.zeros(2,1, 50)
#     x = torch.randn(2,1,12)
#     h_s= gru(x, h_t)
#     print(f'{h_s.size()} \n {h_s}')
#     # print(f'hs: {h_s.size()}, cs: {c_s.size()}')