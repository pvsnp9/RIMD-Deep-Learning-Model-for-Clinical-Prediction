import torch
import math
import torch.nn as nn
from torch.nn.modules.rnn import GRU
from .gru_d import GRUD

class MIMICGRUDModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = args['hidden_size']
        self.grud = GRUD(args['input_size'], args['hidden_size'])
        self.linear = nn.Linear(args['hidden_size'] + args['static_features'], 1)
        self.loss = nn.BCELoss()

    def forward(self,x,statics, mask, delta, x_last_observed, x_mean, y = None):
        x = x.float()
        statics =statics.float()
        mask = mask.float()
        delta = delta.float()
        x_last_observed = x_last_observed.float()
        x_mean = x_mean.float()

        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)

        # split data [batch, 1 X timestamp, static_features] 
        x_s = torch.split(x,1,1)
        mask_s = torch.split(mask, 1, 1)
        delta_s = torch.split(delta, 1, 1)
        x_last_observed_s = torch.split(x_last_observed, 1, 1)

        # get hs for each time stamp from RIMs cell
        for xs, ms, ds, xl_s in zip(x_s, mask_s, delta_s, x_last_observed_s):
            hs = self.grud(torch.squeeze(xs), torch.squeeze(ms), torch.squeeze(ds), torch.squeeze(xl_s), torch.squeeze(x_mean), hs)

        hs = hs.contiguous().view(x.size(0), -1)

        #concatenate static features 
        full_data = torch.cat([hs, statics], dim=1)

        #FCN
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

class MIMICGRUDLosModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = args['hidden_size']
        self.grud = GRUD(args['input_size'], args['hidden_size'])
        self.linear = nn.Linear(args['hidden_size'] + args['static_features'], 1)

    def forward(self,x,statics, mask, delta, x_last_observed, x_mean, y = None):
        x = x.float()
        statics =statics.float()
        mask = mask.float()
        delta = delta.float()
        x_last_observed = x_last_observed.float()
        x_mean = x_mean.float()

        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)

        # split data [batch, 1 X timestamp, static_features] 
        x_s = torch.split(x,1,1)
        mask_s = torch.split(mask, 1, 1)
        delta_s = torch.split(delta, 1, 1)
        x_last_observed_s = torch.split(x_last_observed, 1, 1)

        # get hs for each time stamp from RIMs cell
        for xs, ms, ds, xl_s in zip(x_s, mask_s, delta_s, x_last_observed_s):
            hs = self.grud(torch.squeeze(xs), torch.squeeze(ms), torch.squeeze(ds), torch.squeeze(xl_s), torch.squeeze(x_mean), hs)

        hs = hs.contiguous().view(x.size(0), -1)

        #concatenate static features 
        full_data = torch.cat([hs, statics], dim=1)

        #FCN
        predictions = self.linear(full_data)
        return predictions

    def to_device(self, x):
        return torch.from_numpy(x).to(self.device)