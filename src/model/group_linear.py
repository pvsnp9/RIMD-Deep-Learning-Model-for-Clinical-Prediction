import torch
import torch.nn as nn 
import math


class GroupLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks, bias=True):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.num_blocks = num_blocks

        self.weight = nn.Parameter(torch.Tensor(num_blocks, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', num_blocks=' + str(self.num_blocks) \
            + ', bias=' + str(self.bias is not None) + ')'

    def forward(self, x):
        x =  x.permute(1,0,2)
        x = torch.bmm(x,self.weight) 
        x = x.permute(1,0,2)
        
        if self.bias is not None:
            x += self.bias

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
class CustomLinear(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_features = in_feature
        self.out_features = out_feature
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        assert self.in_features == x.size()[-1], f'Use tensor with {self.in_features} Input Features'
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
'''
weights = uniform random distribution
size = [num_blocks, input, output]

forward:
permute data with num_blocks and perform bmm with given data and weights
return [batch_size, num_blocks, output]
'''
# if __name__ == '__main__':
#     d = GroupLinear(2, 5,3, False)
#     x = torch.randn(2,3,2)
#     print(d(x))