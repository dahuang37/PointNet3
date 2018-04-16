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

        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        #self.mp1 = torch.nn.MaxPool1d(2048)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 40)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        x = torch.max(x,2)[0]


        #x = x.view(-1, 1024)
        x = F.relu(self.bn5(self.fc1(x)))

        x = self.fc2(x)

        return x
