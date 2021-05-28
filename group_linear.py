import torch 
import torch.nn as nn 

class GroupLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(num_blocks, input_size, output_size))

    def forward(self, x):
        x =  x.permute(1,0,2)
        x = torch.bmm(x,self.weights)
        return x.permute(1,0,2)

'''
weights = uniform random distribution
size = [num_blocks, input, output]

forward:
permute data with num_blocks and perform bmm with given data and weights
return [batch_size, num_blocks, output]
'''