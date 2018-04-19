from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RangeNormalize

def rangeNormalize(data):
    return (data - torch.min(data))*(0.99)/(torch.max(data)-torch.min(data))

def histogram(data, bins, range=(0,1)):
    """
    find the histogram of given data
    Current range, (0,1), others haven't been tested

    return histogram, TODO: ret_index, ret_value

    """
    ret_histogram = torch.zeros(bins)
    ret_indexes = torch.zeros(data.size())
    for idx, d in enumerate(data.view(data.numel())):
        if d == 1.0:
            bin_number = bins - 1
        else:
            bin_number = int(bins*d/(range[1]-range[0]))

        if bin_number == bins:
            bin_number = bin_number - 1

        ret_histogram[bin_number] += 1
        idx_a = int(idx/data.size()[-1])
        idx_b = idx%data.size()[-1]
        ret_indexes[idx_a, idx_b] = bin_number

    return ret_histogram, ret_indexes

def histogram_vectorize(data, bins, range=(0,1)):
    """
    find the histogram of given data: batch x seq x feature
    Current range, (0,1), others haven't been tested

    return histogram, TODO: ret_index, ret_value

    """
    ret_histogram = torch.zeros(bins).cuda()
    ret_indexes = torch.zeros(data.size()).cuda()

    # calculate the the corresponding bin number
    temp = (data*bins).int()

    # transform into one-hot vector and then sum
    y_tensor = temp.clone()
    y_tensor = y_tensor.type(torch.LongTensor).view(data.size()[0],-1, 1)
    # given a list of numbers, we transform each number {i} to one_hot vector(nbins x 1) where index i is 1
    y_one_hot = torch.zeros(data.size()[0],y_tensor.size()[1], bins).cuda().scatter_(2, y_tensor, 1)

    ret_indexes = temp.view(data.size())
    ret_histogram = y_one_hot.sum(1)

    return ret_histogram, ret_indexes

def histogram2d_vectorize(data, bins, range=(0,1)):
    """
    find the histogram of given data: batch x seq x feature
    Current range, (0,1), others haven't been tested

    return histogram along the second column. eg : batch x bins x feature

    """
    ret_histogram = torch.zeros(bins).cuda()
    ret_indexes = torch.zeros(data.size()).cuda()

    # calculate the the corresponding bin number
    temp = (data*bins).int()
    temp = temp.transpose(2,1)
    # transform into one-hot vector and then sum
    y_tensor = temp.clone()
    y_tensor = y_tensor.type(torch.LongTensor).view(data.size()[0], data.size()[-1],-1, 1)

    # given a list of numbers, we transform each number {i} to one_hot vector(nbins x 1) where index i is 1
    # scatter(dim, index, val)
    y_one_hot = torch.zeros(data.size()[0], data.size()[-1], data.size()[1], bins).scatter_(3, y_tensor.long(), 1)

    ret_indexes = temp.transpose(2,1)
    ret_histogram = y_one_hot.sum(2).transpose(2,1)

    return ret_histogram, ret_indexes

class HistogramFunction(torch.autograd.Function):
    """
    assume input is already normalized from 0 to 1
    assume input size is batch_size, num_points, feature_size

    """

    """
    PARAM: input, weight, bins

    input: batch_size, num_points, feature_size
    weight:  (TBD) size of input OR size of bins
    bins: 10 by defaulted
    """
    @staticmethod
    def forward(ctx, input, weight, bins=10):
        batch_size = input.size()[0]
        # histograms = torch.zeros(batch_size, bins).cuda()
        # histograms_indexes = torch.zeros(input.size()).cuda()

        histograms, histograms_indexes = histogram_vectorize(input, bins=bins)
        histograms = histograms.cuda()
        histograms_indexes = histograms_indexes.cuda()
        histograms = histograms* weight

        #non-vec version
        # for i in range(batch_size):
        #     # assert sum is the same as the count
        #     histograms[i,:], histograms_indexes[i,:] = histogram(input[i], bins=bins)
        #     histograms[i,:] = histograms[i,:] * weight

        ctx.save_for_backward(input, weight)
        ctx.histograms_indexes = histograms_indexes
        ctx.histograms = histograms
        return histograms

    @staticmethod
    def backward(ctx, grad_output):
        input, weight= ctx.saved_tensors
        histograms_indexes = ctx.histograms_indexes
        histograms = ctx.histograms

        grad_input = torch.zeros(input.size()).cuda()

        """
        the gradient at batch item i, point a, element b:
          grad_output[i, bin_number]*weight[bin_number]/(LEN(histogram[i,bin_number]))
        """
        values = grad_output.data*weight/histograms
        for i in range(input.size()[0]): #batch_size
            grad_input[i] = torch.index_select(values[i], 0, histograms_indexes[i].long().view(-1)).view(grad_input[i].size())

        # for i in range(input.size()[0]): #batch_size
        #     for a in range(input.size()[1]): #num_of_points
        #         for b in range(input.size()[2]):
        #             bin_number = int(histograms_indexes[i,a,b])
        #
        #             grad_input[i,a,b] = grad_output.data[i,bin_number]*weight[bin_number]/histograms[i,bin_number] #TODO: divide by number of element in the bin
        bs, seq, fs, bn = input.size()[0], input.size()[1], input.size()[2], weight.size()[0]
        # np equivalent
#         idx_cat = torch.from_numpy(np.repeat(np.arange(bs),seq*fs)*bn).view(indices.shape)
        # torch version
        idx_cat = torch.arange(0, bs*bn, step=bn).cuda().unsqueeze(-1).expand(-1,seq*fs)

        idx_cat = idx_cat.view(histograms_indexes.shape).int()

        histograms_indexes = histograms_indexes + idx_cat

        grad_input = torch.index_select(values.view(-1), 0, histograms_indexes.long().view(-1)).view(bs,seq,fs)


        grad_weight = (grad_output.data*histograms).sum(0)/bs
        """

        """
        grad_weight = (grad_output.data*histograms.sum(0)).sum(0)/input.size()[0]



        return Variable(grad_input), Variable(grad_weight), None


class Histogram(nn.Module):
    def __init__(self, bins=10):
        super(Histogram, self).__init__()
        self.bins = bins

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weights = nn.Parameter(torch.Tensor(bins).cuda())

        # Not a very smart way to initialize weights
        self.weights.data.uniform_(0, 1)


    def forward(self, input):

        return HistogramFunction.apply(input, self.weights, self.bins)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'bins={}'.format(self.bins)

class Histogram2dFunction(torch.autograd.Function):
    """
    assume input is already normalized from 0 to 1
    assume input size is batch_size, num_points, feature_size

    """

    """
    PARAM: input, weight, bins

    input: batch_size, num_points, feature_size
    weight:  (TBD) size of input OR size of bins
    bins: 10 by defaulted
    """
    @staticmethod
    def forward(ctx, input, weight, bins=128):
        histograms, histograms_indexes = histogram2d_vectorize(input, bins=bins)

        histograms = histograms.cuda()
        histograms_values = histograms_indexes.cuda()
        # hist size: batch, bins, feature; weight size: bins, feature
        histograms = histograms*weight

        ctx.save_for_backward(input, weight)
        ctx.histograms_indexes = histograms_indexes
        ctx.histograms = histograms
        return histograms

    @staticmethod
    def backward(ctx, grad_output):
        input, weight= ctx.saved_tensors
        histograms_indexes = ctx.histograms_indexes
        histograms = ctx.histograms

        grad_input = torch.zeros(input.size()).cuda()

        """
        the gradient at batch item i, point a, element b:
          grad_output[i, bin_number]*weight[bin_number]/(LEN(histogram[i,bin_number]))
        """
        values = grad_output.data*weight/histograms
#         for i in range(input.size()[0]): #batch_size
#             grad_input[i] = torch.index_select(values[i], 0, histograms_values[i].long().view(-1)).view(grad_input[i].size())

#             for a in range(input.size()[1]): #num_of_points
#                 for b in range(input.size()[2]):
#                     bin_number = int(histograms_values[i,a,b])

#                     grad_input[i,a,b] = grad_output.data[i,bin_number]*weight[bin_number]/histograms[i,bin_number] #TODO: divide by number of element in the bin
        # values = grad_output
        # histogram_indexes = indices
        bs, seq, fs, bn = input.size()[0], input.size()[1], input.size()[2], weight.size()[0]

        histograms_indexes = histograms_indexes.transpose(2,1)
        # np version
#         i2 = torch.from_numpy(np.repeat(np.arange(bs*fs),seq)*bn).view(indices.shape)
        # torch version
        idx_cat = torch.arange(0, bs*fs*bn, step=bn).cuda().unsqueeze(-1).expand(-1,seq)
        idx_cat = idx_cat.contiguous().view(histograms_indexes.shape).int()
        # batch x feature x bins
        values = values.transpose(2,1)
        histograms_indexes = histograms_indexes + idx_cat

        grad_input = torch.index_select(values.contiguous().view(-1), 0, histograms_indexes.long().contiguous().view(-1)).view(histograms_indexes.shape)

        grad_input = grad_input.transpose(2,1)

        grad_weight = (grad_output.data*histograms).sum(0)/bs


        return Variable(grad_input), Variable(grad_weight), None


class Histogram2d(nn.Module):
    def __init__(self, feature_size, bins=10):
        super(Histogram2d, self).__init__()
        self.bins = bins
        self.feature_size = feature_size

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weights = nn.Parameter(torch.Tensor(bins, feature_size).cuda())

        # Not a very smart way to initialize weights
        self.weights.data.fill_(1)


    def forward(self, input):

        return Histogram2dFunction.apply(input, self.weights, self.bins)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'bins={}'.format(self.bins)

class Test(nn.Module):
    def __init__(self, input_dim, maxout, bins=512):
        super(Test, self).__init__()
        self.maxout = maxout
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        # self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)

        self.his = Histogram2d(128, bins=bins)

        self.fc1 = nn.Linear(bins*128, 256)
        self.fc2 = nn.Linear(256, 40)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.bn3(self.conv3(x))

        x = x.transpose(2,1)
        x = rangeNormalize(x)
        x = self.his(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        out = self.fc2(x)

        return out
