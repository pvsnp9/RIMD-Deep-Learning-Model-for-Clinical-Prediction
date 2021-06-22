import math
import torch 
import torch.nn as nn
from .rim import RIMCell

class MIMICModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rim_cell = RIMCell(
            args['input_size'], args['hidden_size'], args['num_rims'],
            args['active_rims'], args['rnn_cell'], args['input_key_size'],
            args['input_value_size'], args['input_query_size'],
            args['num_input_heads'], args['input_dropout'],
            args['comm_key_size'], args['comm_value_size'], args['comm_query_size'],
            args['num_comm_heads'], args['comm_dropout']
        )

        # TODO: Add more statice variables to linear layer
        self.linear_one = nn.Linear(args['hidden_size'] * args['num_rims'] + args['static_features'], 10)
        self.linear_two = nn.Linear(10, 1)

        self.loss = nn.BCELoss()

    def forward(self, x, static, y = None):
        x = x.float()
        static = static.float()

        # initialize hidden state
        hs = torch.zeros(x.size(0), self.args['num_rims'], self.args['hidden_size']).to(self.device)

        if self.args['rnn_cell'] == 'LSTM':
            cs = torch.zeros(x.size(0), self.args['num_rims'], self.args['hidden_size']).to(self.device)
        else:
            cs = None
        # split data [batch, 1 X timestamp, static_features] 
        splitted_ts = torch.split(x,1,1)

        # get hs,cs for each time stamp from RIMs cell
        for ts in splitted_ts:
            hs,cs = self.rim_cell(ts, hs, cs)
        
        hs = hs.contiguous().view(x.size(0), -1)
        # concate static features
        full_data = torch.cat([hs, static], dim=1)
        # FCN 
        l1 = self.linear_one(full_data)
        predictions = self.linear_two(l1)

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