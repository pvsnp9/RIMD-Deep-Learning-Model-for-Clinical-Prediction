import torch 
import torch.nn as nn 

class GroupLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.num_blocks = num_blocks

        self.weights = nn.Parameter(0.01 * torch.randn(num_blocks, input_size, output_size))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', num_blocks=' + str(self.num_blocks) \
            + ', bias=' + str(None) + ')'

    def forward(self, x):
        x =  x.permute(1,0,2)
        x = torch.bmm(x,self.weights)
        return x.permute(1,0,2)

    
    # def forwa
'''
weights = uniform random distribution
size = [num_blocks, input, output]

forward:
permute data with num_blocks and perform bmm with given data and weights
return [batch_size, num_blocks, output]
'''
