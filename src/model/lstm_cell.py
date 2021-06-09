import torch 
import torch.nn as nn
import math

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.Tensor(input_size,hidden_size *4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, hidden_state):
        # X [batch_size, time_stamp, features] time_stamp = 1 squeezed
        x = x.squeeze()
        h_t, c_t = hidden_state

        gates = x @ self.W + h_t @ self.U + self.bias

        i_t = torch.sigmoid(gates[:, :self.hidden_size]) #input
        f_t = torch.sigmoid(gates[:, self.hidden_size: self.hidden_size *2]) # forget
        g_t = torch.tanh(gates[:, self.hidden_size * 2 : self.hidden_size *3]) # 
        o_t = torch.sigmoid(gates[:, self.hidden_size * 3 :]) # output

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

# if __name__ == '__main__':
#     lstm = CustomLSTMCell(12, 50)
#     h_t =torch.zeros(2, 50)
#     c_t = torch.zeros(2, 50)
#     x = torch.randn(2,1,12)
#     h_s, c_s = lstm(x, (h_t, c_t))
#     print(f'hs: {h_s.size()}, cs: {c_s.size()}')
    # v = torch.randn(2,1,10)
    # v= v.squeeze()
    # # vs = v[:,1,:]
    # print(f'shape: {v.size()}')
