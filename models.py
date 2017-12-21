from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        out = self.linear(x)
        out = F.dropout(out, training=self.training)
        return out

class STN(nn.Module):
    def __init__(self, num_points = 2048):
        super(STN, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv2d(3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool2d((num_points,1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)))#.view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class LogisticRegressionWithStn(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionWithStn, self).__init__()
        self.stn = STN()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = x.view(4, 2048, 3)
        x = torch.bmm(x, trans)
        
        
        #print(x.view(4,2048*3).size())
        out = self.linear(x.view(-1, 2048*3))
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.out = nn.Linear(128, 40)

    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(2,1)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

class LSTM_dropout(nn.Module):
    def __init__(self):
        super(LSTM_dropout, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.out = nn.Linear(128, 40)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(2,1)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out