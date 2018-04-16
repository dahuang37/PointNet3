from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class LSTM_mlp(nn.Module):
    def __init__(self, input_dim, maxout, mlp=[64], fc=[128,40]):
        super(LSTM_mlp, self).__init__()
        self.maxout = maxout
        self.input_dim = input_dim
        self.mlp = torch.nn.Sequential()

        self.mlp.add_module("conv_1", nn.Conv1d(input_dim, mlp[0], 1))
        self.mlp.add_module("bn_1", nn.BatchNorm1d(mlp[0]))
        self.mlp.add_module("relu_1", nn.ReLU())
        
        for i in range(len(mlp)-1):
            self.mlp.add_module("conv"+str(i+2), torch.nn.Conv1d(mlp[i], mlp[i+1], 1))
            self.mlp.add_module("bn"+str(i+2), nn.BatchNorm1d(mlp[i+1]))
            self.mlp.add_module("relu"+str(i+2), nn.ReLU())

        self.rnn = nn.LSTM(input_size=mlp[-1], hidden_size=fc[0], num_layers=2, batch_first=True)
        
        self.fc = torch.nn.Sequential()
        for i in range(len(fc)-2):
            self.fc.add_module("fc_"+str(i+1), nn.Linear(fc[i], fc[i+1]))
            self.fc.add_module("bn_"+str(i+1), nn.BatchNorm1d(fc[i+1]))
        self.fc.add_module("fc_"+str(len(fc)-1), nn.Linear(fc[-2], fc[-1]))

    def forward(self, x):
        print(self.mlp)
        print(self.fc)
        print(self.rnn)
        x = x.transpose(2,1)
        x = self.mlp.forward(x)
        x = x.transpose(2,1)

        self.rnn.flatten_parameters()
        r_out, (h_n, h_c) = self.rnn(x, None)
        
        if self.maxout == 1:
            x = torch.max(r_out, 1)[0]
        else:
            x = r_out[:,-1,:]

        out = self.fc.forward(x)

        return out
