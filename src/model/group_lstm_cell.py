import math
import torch
import torch.nn as nn
from .group_linear import GroupLinear


class GroupLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstms):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstms = num_lstms

        self.input_hidden = GroupLinear(self.input_size, 4 * self.hidden_size, self.num_lstms )
        self.hidden_hidden = GroupLinear(self.hidden_size, 4 * self.hidden_size, self.num_lstms)
        #self.reset_parameters()
    
    def forward(self, x, hidden_states):
        hs, cs = hidden_states
        input_hidden = self.input_hidden(x)
        hidden_hidden = self.hidden_hidden(hs)

        preact_data = input_hidden + hidden_hidden

        gates = torch.sigmoid(preact_data[:,:,:3 * self.hidden_size])
        candidate_context = torch.tanh(preact_data[:,:,3 * self.hidden_size:])
        input_gate = gates[:,:,:self.hidden_size]
        forget_gate = gates[:,:,self.hidden_size: 2*self.hidden_size]
        out_gate = gates[:,:, -self.hidden_size:]

        next_cell_state = torch.mul(cs, forget_gate) + torch.mul(input_gate, candidate_context)
        next_hidden_state = torch.mul(out_gate, torch.tanh(next_cell_state))

        return next_hidden_state, next_cell_state


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


'''
This can compute the operations of N-LSTMCells at once.
forward:
    - input: X => [batch_size, num_lstms, input_size]
             hidden_sates = (hs,cs) => [batch_size, num_lstms, hidden_state_size]
    - output: hs, cs => [batch_size, num_lstms, hidden_state_size]
'''