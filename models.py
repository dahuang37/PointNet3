from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import lstm_models as lm
# from lstm_models import LSTM, LSTMCell

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
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.rnn = nn.LSTM(input_size=1024, hidden_size=128, num_layers=1, batch_first=True)
        self.out = nn.Linear(128, 40)

    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.transpose(2,1)
        self.rnn.flatten_parameters()
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

class LSTM_dist(nn.Module):
    def __init__(self, input_dim=3, batch_size=32, hidden_size=128):
        super(LSTM_dist, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.rnn = nn.LSTMCell(input_size=64, hidden_size=self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 40)
        
    def init_hidden(self, batch_size):
        dtype = torch.cuda.FloatTensor
        # hx = torch.randn(self.batch_size, self.hidden_size).type(dtype)
        # cx = torch.randn(self.batch_size, self.hidden_size).type(dtype)
        hx = torch.randn(batch_size, self.hidden_size).type(dtype)
        cx = torch.randn(batch_size, self.hidden_size).type(dtype)
        return Variable(hx), Variable(cx)

    def forward(self, x, batch_size=32):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(2,1)
        #hx = Variable(x.data.new(1, 1, 128).zero_())
        #hx = Variable(x.data.new(x.size()[0], 128).zero_())
        #h_x, c_x = (hx, hx) # self.hidden <- this cuases memory error
        h_x, c_x = self.init_hidden(batch_size)
        output = []
        for i in range(x.size()[1]):
            # print("sequence size:")
            #print(x[:,i].unsqueeze(1).size())
            # print(x.size())
            #print(x[:,i].size())
            h_x, c_x = self.rnn(x[:,i], (h_x, c_x))
            output.append(h_x)
        output = torch.stack(output, 1).squeeze(2)
        # print(output.size())
        out = self.out(output[:, -1, :])
        return out

class LSTM_dist_custom(nn.Module):
    def __init__(self):
        super(LSTM_dist_custom, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.rnn = lm.LSTM(cell_class=lm.LSTMCell, input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.out = nn.Linear(128, 40)

    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(2,1)
        r_out, (h_n, h_c) = self.rnn(x, None)
        # print("output size")
        # print(r_out[:,-1,:].size())
        out = self.out(r_out[:, -1, :])
        # print("out size")
        # print(out.size())
        return out

class RNN_test(nn.Module):
    """
    idea: run through the seq to get the best h-x, use that as the next input memeory cell

    """
    def __init__(self):
        super(RNN_test, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.hidden_size = 128

        self.inp = nn.Linear(3, 64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.out1 = nn.Linear(256,128)
        self.out = nn.Linear(128,40)

    def init_hidden(self, batch_size):
        dtype = torch.cuda.FloatTensor
        # hx = torch.randn(self.batch_size, self.hidden_size).type(dtype)
        # cx = torch.randn(self.batch_size, self.hidden_size).type(dtype)
        hx = torch.randn(1, batch_size, self.hidden_size).type(dtype)
        cx = torch.randn(1, batch_size, self.hidden_size).type(dtype)
        return Variable(hx), Variable(cx)

    def step(self, input, hidden=None):
        input = self.inp(input).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, batch_size=32, hidden=None, steps=0):
        # x = x.transpose(2,1)
        # x = F.relu(self.bn1(self.conv1(x)))
        # # x = F.relu(self.bn2(self.conv2(x)))
        # # x = F.relu(self.bn3(self.conv3(x)))
        
        # x = x.transpose(2,1)
        if steps == 0: steps = inputs.size()[1]
        outputs = Variable(torch.zeros(steps, batch_size, 40).cuda())
        (hx, cx) = self.init_hidden(batch_size)
        #print(steps)
        for i in range(steps):
            input = inputs[:,i]   
            output, (hx_new, cx_new) = self.step(input, (hx, cx))
            hx, cx = (hx, cx) if torch.gt(hx, hx_new).all() else hx_new, cx_new
            outputs[i] = output
        #print(outputs[-1,:,:])
        # r_out, h_n = self.rnn(x, None)
        # # out1 = self.out1(r_out[:, -1, :])
        # # print(r_out.size())
        # # print(r_out.squeeze(1).size())
        # # print(r_out[:,-1,:].size())
        # out = self.out(r_out[:,-1,:])
        # # print(out.size())
        
        return outputs[-1,:,:]
