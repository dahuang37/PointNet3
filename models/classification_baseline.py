from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Baseline(nn.Module):
    def __init__(self, input_dim, maxout):
        super(Baseline, self).__init__()
        self.maxout = maxout
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc3 = nn.Linear(128, 40)
 
    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
       
        x = x.transpose(2,1)
        r_out, (h_n, h_c) = self.rnn(x, None)
        
        if self.maxout == 1:
            x = torch.max(r_out, 1)[0]
        else:
            x = r_out[:,-1,:]
    
        out = self.fc3(x)
        return out