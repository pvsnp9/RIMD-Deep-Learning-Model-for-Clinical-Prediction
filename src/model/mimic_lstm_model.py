import torch 
import torch.nn as nn
from .lstm_cell import CustomLSTMCell
import math

class MIMICLSTMModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.hidden_size = args['hidden_size']
        self.lstm = CustomLSTMCell(args['input_size'], args['hidden_size'])
        self.linear = nn.Linear(args['hidden_size'] + args['static_features'], 1)
        self.loss = nn.BCELoss()


    def forward(self, x, static, y = None):
        x = x.float()
        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)
        cs = torch.randn(x.size(0), self.hidden_size).to(self.device)

        # split data [batch, 1 X timestamp, static_features] 
        splitted_ts = torch.split(x,1,1)
        # get hs,cs for each time stamp from RIMs cell
        for ts in splitted_ts:
            hs,cs = self.lstm(ts, (hs, cs))
        
        hs = hs.contiguous().view(x.size(0), -1)

        # concatenate static features 
        full_data = torch.cat([hs, static], dim=1)
        # FCN
        predictions = self.linear(full_data)

        if y is not None:
            y = y.float()
            probs = torch.sigmoid(predictions)
            loss = self.loss(probs.view(-1), y)
            return probs, loss
        
        return predictions

    
    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()
        total_norm = math.sqrt(total_norm)
        return total_norm
    
    def to_device(self, x):
        return torch.from_numpy(x).to(self.device) 


class MIMICLSTMLosModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.hidden_size = args['hidden_size']
        self.lstm = CustomLSTMCell(args['input_size'], args['hidden_size'])
        self.linear = nn.Linear(args['hidden_size'] + args['static_features'], 1)
    
    def forward(self, x, static, y = None):
        x = x.float()
        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)
        cs = torch.randn(x.size(0), self.hidden_size).to(self.device)

        # split data [batch, 1 X timestamp, static_features] 
        splitted_ts = torch.split(x,1,1)
        # get hs,cs for each time stamp from RIMs cell
        for ts in splitted_ts:
            hs,cs = self.lstm(ts, (hs, cs))
        
        hs = hs.contiguous().view(x.size(0), -1)

        # concatenate static features 
        full_data = torch.cat([hs, static], dim=1)
        # FCN
        predictions = self.linear(full_data)        
        return predictions

    def to_device(self, x):
        return torch.from_numpy(x).to(self.device) 