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
from models import LSTM_dist, LogisticRegression, LSTM_dist_custom, LSTM, RNN_test
from logger import Logger
from utils import Random_permute, Rotate_point_cloud, Jitter_point_cloud


# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='model to train')
parser.add_argument('--learning_rate', type=float, default='0.01', help='learning rate')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--sort', type=int, default = 0,  help='sort input, 0 not sorted, 1 sorted')
parser.add_argument('--distance', type=int, default = 0,  help='expand dim to include distance between points')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--outf', type=str, default='logs',  help='output folder')
parser.add_argument('--path', type=str, default = '',  help='model path')
parser.add_argument('--debug', type=int, default = 0,  help='debug mode, not creating saved path')
parser.add_argument('--random_input', type=int, default = 0,  help='if 1, input will be randomized for training')

opt = parser.parse_args()
print(opt)

# create logger and folder for savings
if(opt.debug == 0):
    save_path = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_path = opt.outf + "/" + opt.model + "/" + save_path
    train_path = save_path + "/train"
    test_path = save_path + "/test"

    try:
        os.makedirs('%s' % train_path)
        os.makedirs('%s' % save_path)
    except OSError:
        pass

    with open(save_path+'/hyperparamters.txt', 'w') as file:
        for arg in vars(opt):
            sentence = arg + " " + str(getattr(opt,arg)) + "\n"
            print(arg, getattr(opt, arg))
            file.write(sentence)
        file.close()

    # set up logger for loss, accuracy graph
    train_logger = Logger(train_path)
    test_logger = Logger(test_path)

# randomize seed
opt.manualSeed = random.randint(1, 10000) # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# load data
root = "data/modelnet40_ply_hdf5_2048/"
use_cuda = torch.cuda.is_available()

# parameters for dataset
sort_input = True if opt.sort == 1 else False
distance_input = True if opt.distance == 1 else False

# load transformations
if opt.random_input == 0:
    train_transform = transforms.Compose([Rotate_point_cloud(), Jitter_point_cloud()]) # add random later
else:
    train_transform = transforms.Compose([Random_permute(opt.num_points), Rotate_point_cloud(), Jitter_point_cloud()])
test_transform = transforms.Compose([Rotate_point_cloud(), Jitter_point_cloud()])
# Load dataset / data loader
train_dataset = ModelNetDataset(root, 
                        train=True, 
                        sort=sort_input,
                        transform=train_transform,
                        distance=distance_input)
train_loader = DataLoader(train_dataset, 
                        batch_size=opt.batchSize,
                        shuffle=True, 
                        num_workers=opt.workers)

test_dataset = ModelNetDataset(root, 
                        train=False, 
                        sort=sort_input, 
                        transform=test_transform,
                        distance=distance_input)
test_loader = DataLoader(test_dataset, 
                        batch_size=opt.batchSize,
                        shuffle=True, 
                        num_workers=opt.workers)




# define model
if opt.model == 'lstm':
    model = LSTM(input_dim=3)
elif opt.model == 'lstm_dist':
    model = LSTM_dist(input_dim=3)
elif opt.model == 'test':
    model = RNN_test()
elif opt.model == "logit":
    model = LogisticRegression(2048*3, 40)

# load speicified pre-trained model
if opt.path != '':
    model.load_state_dict(torch.load(opt.path))

# define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# transfer model and criterion to cuda if exist
if use_cuda:
    model = nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

# initilize variables for results
since = time.time()
best_model_wts = model.state_dict()
best_acc = 0.0
best_epoch = 0

# early stopping 
'''
monitor on val loss
if stop improving from the past 10 epoch, then stop
'''
patience = 10
wait = 0
current_best = np.inf
stop_training = False

for epoch in range(opt.nepoch):

    # training
    correct = 0.0
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # get mini-batch data
        if use_cuda:
            data, target = data.cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target[:,0])
        target = target.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # feedforward
        output = model(data)

        # compute loss
        loss = criterion(output, target)

        # compute the gradients
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-opt.learning_rate, p.grad.data)

        # optimize / backprop
        optimizer.step()

        # get loss and accuracy
        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # logging
        if batch_idx % 100 == 0:
            current_size = data.size()[0] if batch_idx == 0 else batch_idx*data.size()[0]
            train_loss /= current_size
            accuracy = correct / current_size

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t Accuracy: {:.2f}%'.format(\
                epoch, batch_idx * len(data), len(train_loader.dataset),\
                100. * batch_idx / len(train_loader), train_loss, accuracy*100.))
            if(opt.debug == 0):
                info = {
                    'loss': train_loss,
                    'accuracy': accuracy
                }
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, epoch*len(train_loader.dataset)+batch_idx*data.size()[0])

    # eval
    test_loss = 0.0
    correct = 0.0
    model.eval()
    for (data, target) in test_loader:
        # get mini-batch data
        data, target = data.cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target[:,0])
        target = target.long()

        # feedforward
        output = model(data)

        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # logging
    test_loss /= len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100.*epoch_acc))

    if(opt.debug == 0):
        info = {
            'loss': test_loss,
            'accuracy': epoch_acc
        }
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, epoch*len(train_loader.dataset))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()
        best_epoch = epoch

    # early stopping
    if current_best > test_loss:
        current_best = test_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            stop_training = True

    if stop_training:
        break


# saves training info
time_elapsed = time.time() - since
print_time = 'Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60)
print(print_time)

print('Best val Acc: {:2f}'.format(100.*best_acc))

if(opt.debug == 0):
    # saving the model and training info
    torch.save(best_model_wts, '%s/model.pth' % (save_path))
    with open(save_path+'/result.txt', 'w') as file:
        if stop_training:
            print_stop_training = "Stop early at " + str(epoch) + " \n"
            file.write(print_stop_training)
        print_best_accuracy = "Best Accuracy: " + str(100.*best_acc) + " at " + str(best_epoch) +  "th epoch \n"
        file.write(print_best_accuracy)
        file.write(print_time)
        file.close()