# RIM and cell codes are taken from user with modification

from src.model.group_gru_cell import GroupGRUCell
import torch
import torch.nn as nn
import math
import numpy as np
from .group_linear import GroupLinear, CustomLinear
from .group_lstm_cell import GroupLSTMCell
from .blocked_gradients import BlockedGradients

class RIMCell(nn.Module):
    def __init__(self,
        input_size, hidden_size, num_rims,
        active_rims, rnn_cell,
        input_key_size = 64, input_value_size = 400, input_query_size=64,
        num_input_heads=1, input_dropout = 0.1, comm_key_size = 32, 
        comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4,
        comm_dropout = 0.1

    ):
        super().__init__()
        assert comm_value_size == hidden_size, "RIM Communication values size must be equal with hidden Size"
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rims = num_rims
        self.active_rims = active_rims
        self.rnn_cell = rnn_cell

        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_values_size = input_value_size
        self.num_input_heads = num_input_heads

        self.comm_key_size = comm_key_size
        self.comm_query_size= comm_query_size
        self.comm_value_size = comm_value_size
        self.num_comm_heads = num_comm_heads

        self.input_key_layer = CustomLinear(self.input_size, self.num_input_heads* self.input_query_size, bias=False)
        self.input_value_layer = CustomLinear(self.input_size, self.num_input_heads * self.input_values_size, bias=False)
        self.input_dropout = nn.Dropout(p=input_dropout)

        if self.rnn_cell == 'LSTM':
            self.rnn = GroupLSTMCell(self.input_values_size, self.hidden_size,  self.num_rims)
            self.input_query_layer = GroupLinear(self.hidden_size, self.input_key_size * self.num_input_heads, self.num_rims, bias=False)
        else:
            # GRU
            self.rnn = GroupGRUCell(self.input_values_size, self.hidden_size,  self.num_rims)
            self.input_query_layer = GroupLinear(self.hidden_size, self.input_key_size * self.num_input_heads, self.num_rims, bias=False)
        
        self.comm_key_layer = GroupLinear(self.hidden_size,self.comm_key_size * self.num_comm_heads, self.num_rims, bias=False)
        self.comm_value_layer = GroupLinear(self.hidden_size, self.comm_value_size * self.num_comm_heads, self.num_rims, bias=False)
        self.comm_query_layer = GroupLinear(self.hidden_size, self.comm_query_size * self.num_comm_heads, self.num_rims, bias=False)

        self.comm_attention_output = GroupLinear(self.num_comm_heads * self.comm_value_size, self.comm_value_size,self.num_rims, bias=False)
        self.comm_dropout = nn.Dropout(p=comm_dropout)


    def transpose_for_score(self, x, num_attention_head, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_head, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def input_attention_mask(self, x, h):
        """
	    Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
	    		h (batch_size, num_units, hidden_size)
	    Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
	    		mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
		"""

        key_layer = self.input_key_layer(x)
        value_layer = self.input_value_layer(x)
        query_layer = self.input_query_layer(h)

        key_layer = self.transpose_for_score(key_layer, self.num_input_heads, self.input_key_size)
        query_layer = self.transpose_for_score(query_layer, self.num_input_heads, self.input_query_size)
        value_layer = self.transpose_for_score(value_layer, self.num_input_heads, self.input_values_size)
        value_layer = torch.mean(value_layer, dim=1)

        attention_score = torch.matmul(query_layer, key_layer.transpose(-1,-2)) / math.sqrt(self.input_key_size)
        attention_score = torch.mean(attention_score, dim=1)
        mask = torch.zeros(x.size(0), self.num_rims).to(self.device)

        signal_attention = attention_score[:,:,0]
        topk1 = torch.topk(signal_attention, self.active_rims, dim=1)
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.active_rims)

        mask[row_index, topk1.indices.view(-1)] = 1

        distributed_probs = nn.Softmax(dim=-1)(attention_score)
        attention_probs = self.input_dropout(distributed_probs)

        inputs = torch.matmul(attention_probs, value_layer) * mask.unsqueeze(2)

        return inputs, mask

    def communication_attention(self, input_hidden, mask):
        """
	    Input : input_hidden (batch_size, num_units, hidden_size)
	    	    mask obtained from the input_attention_mask() function
	    Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
	    """

        key_layer = self.comm_key_layer(input_hidden)
        query_layer = self.comm_query_layer(input_hidden)
        value_layer = self.comm_value_layer(input_hidden)

        key_layer = self.transpose_for_score(key_layer, self.num_comm_heads, self.comm_key_size)
        query_layer = self.transpose_for_score(query_layer, self.num_comm_heads, self.comm_query_size)
        value_layer = self.transpose_for_score(value_layer, self.num_comm_heads, self.comm_value_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2)) / math.sqrt(self.comm_key_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        mask = [mask for _ in range(attention_probs.size(1))]
        mask = torch.stack(mask, dim=1)

        attention_probs = attention_probs * mask.unsqueeze(3)
        attention_probs = self.comm_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_contex_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
        context_layer = context_layer.view(*new_contex_layer_shape)
        context_layer = self.comm_attention_output(context_layer)
        context_layer = context_layer + input_hidden

        return context_layer


    def forward(self, x, hs, cs = None):
        """
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
        null_signal_input = torch.zeros(x.size()[0], 1, x.size()[2]).float().to(self.device)
        x = torch.cat((x, null_signal_input), dim=1)

        #compute input attention for each RIM input
        inputs, mask = self.input_attention_mask(x, hs)

        hs_prev = hs
        if cs is not None: cs_prev = cs

        
        #compute hidden and or cell state for communication attention from N-RNN 
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        #Block gradient through inactive rim units
        mask = mask.unsqueeze(2)
        h_new = BlockedGradients.apply(hs, mask)

        # compute communication attention for next hidden state
        hidden_comm = self.communication_attention(h_new, mask.squeeze(2))

        hs = mask * hidden_comm + (1-mask) * hs_prev
        if cs is not None:
            cs = mask * cs + (1-mask) * cs_prev
            return hs, cs
        
        return hs, None


