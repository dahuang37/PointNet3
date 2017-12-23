from __future__ import print_function, division
import os
import time
import argparse
import numpy as np
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from dataset import ModelNetDataset
from models import LSTM, LogisticRegression
from logger import Logger


# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='logs',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--learning_rate', type=float, default='0.01', help='learning rate')

opt = parser.parse_args()
print (opt)

def permute_transoform(data):
    permutations = torch.randperm(2048)
    data_cat = data[permutations]
    
    return data_cat

transform = transforms.Compose([transforms.Lambda(lambda x:permute_transoform(x)),])
# create folder for savings
save_path = datetime.now().strftime('%Y-%m-%d %H:%M')
save_path = opt.outf + "/" + save_path
try:
    os.makedirs('%s' % save_path)
except OSError:
    pass
# set up logger for loss, accuracy graph
logger = Logger(save_path)

# randomize seed
opt.manualSeed = random.randint(1, 10000) # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# load data
root = "data/modelnet40_ply_hdf5_2048/"
use_cuda = torch.cuda.is_available()

train_loader = DataLoader(ModelNetDataset(root, train=True, transform=transform), batch_size=opt.batchSize,
                       shuffle=True, num_workers=opt.workers)
test_loader = DataLoader(ModelNetDataset(root, train=False, transform=transform), batch_size=opt.batchSize,
                       shuffle=True, num_workers=opt.workers)
print('dataset length %d' % len(train_loader.dataset))

# define model
model = LSTM()
if use_cuda:
    model.cuda()

if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

since = time.time()

best_model_wts = model.state_dict()
best_acc = 0.0
for epoch in range(opt.nepoch):
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target[:,0])
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        # preventing nan 
        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        # for p in model.parameters():
        #     p.data.add_(-opt.learning_rate, p.grad.data)
            
        optimizer.step()
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()   

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], accuracy.data[0]))
            
            info = {
                'loss': loss.data[0],
                'accuracy': accuracy.data[0]
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch*len(train_loader.dataset)+batch_idx*len(data))

    # print("\nAccuracy: {:.2f}%".format(correct*100/len(train_loader.dataset)))

# def test():
    test_loss = 0
    correct = 0
    model.eval()
    for (data, target) in test_loader:
        data, target = data.cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target[:,0])

        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    epoch_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  epoch_acc))

    info = {
        'test_loss': test_loss,
        'test_accuracy': correct * 1.0 / len(test_loader.dataset)
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch*len(train_loader.dataset)+batch_idx*len(data))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

print('Best val Acc: {:4f}'.format(best_acc))

torch.save(best_model_wts, '%s/model.pth' % (save_path))


