import math
import torch
import torch.nn as nn
from .rim_d import RIMDCell

class MIMICDecayModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rim_decay_cell = RIMDCell(
            args['input_size'], args['hidden_size'], args['num_rims'], 
            args['active_rims'], args['rnn_cell'], args['mask_size'],
            args['delta_size'], args['input_key_size'],
            args['input_value_size'], args['input_query_size'],
            args['num_input_heads'], args['input_dropout'],
            args['comm_key_size'], args['comm_value_size'], args['comm_query_size'],
            args['num_comm_heads'], args['comm_dropout']
        )

        self.linear = nn.Linear(args['hidden_size'] * args['num_rims'] + args['static_features'], 1)
        self.loss = nn.MSELoss()

    def forward(self, x, statics, mask, delta, x_last_observed, x_mean, y=None):
        x = x.float()
        statics =statics.float()
        mask = mask.float()
        delta = delta.float()
        x_last_observed = x_last_observed.float()
        x_mean = x_mean.float()

        hs = torch.zeros(x.size(0), self.args['num_rims'], self.args['hidden_size']).to(self.device)

        if self.args['rnn_cell'] == 'LSTM':
            cs = torch.zeros(x.size(0), self.args['num_rims'], self.args['hidden_size']).to(self.device)
        else:
            cs = None
        
        # split data [batch, 1 X timestamp, static_features] 
        x_s = torch.split(x,1,1)
        mask_s = torch.split(mask, 1, 1)
        delta_s = torch.split(delta, 1, 1)
        x_last_observed_s = torch.split(x_last_observed, 1, 1)
        for xs, ms, ds, xl_s in zip(x_s, mask_s, delta_s, x_last_observed_s):
            hs, cs = self.rim_decay_cell(xs, ms, ds, xl_s, x_mean, hs, cs)
            
        hs = hs.contiguous().view(x.size(0), -1)

        # concatenate static features
        if statics is not None:
            full_data = torch.cat([hs, statics], dim=1)
        else:
            full_data = hs
            
        predictions = self.linear(full_data)

        if y is not None:
            y = y.float()
            loss = self.loss(predictions.view(-1), y)
            return predictions, loss

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