from __future__ import print_function, division
import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from models import LogisticRegression, LSTM, RNN_test, LSTM_mlp
import utils
import data

def adjust_learning_rate(learning_rate, optimizer, epoch, saver):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 25))
    saver.log_string("learning rate: %f" % (lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def parse_arguement():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', help='model to train')
    parser.add_argument('--learning_rate', type=float, default='0.01', help='learning rate')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--sort', type=int, default = 0,  help='sort input, 0 not sorted, 1 sorted')
    parser.add_argument('--distance', type=int, default = 0,  help='expand dim to include distance between points')
    parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--outf', type=str, default='testing_',  help='output folder')
    parser.add_argument('--path', type=str, default = '',  help='model path')
    parser.add_argument('--debug', type=int, default = 0,  help='debug mode, not creating saved path')
    parser.add_argument('--random_input', type=int, default = 0,  help='if 1, input will be randomized for training')
    parser.add_argument('--clip', type=float, default=4, help='clip for gradient')
    parser.add_argument('--early_stopping', type=int, default=0, help='1 then will early stop')
    parser.add_argument('--transform', type=int, default = 1, help='if 1, input will be transformed')
    parser.add_argument('--elem_max', type=int, default = 0, help='if 1, we max out lstm output')

    opt = parser.parse_args()

    return opt

def train(model, optimizer, criterion, saver, train_loader, epoch, opt):
    correct = 0.0
    train_loss = 0.0
    train_total = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # get mini-batch data
        if opt.transform:
            data = torch.from_numpy(utils.rotate_point_cloud(data.numpy()))
            # data = torch.from_numpy(jitter_point_cloud(data.numpy()))
            
        if torch.cuda.is_available():
            data, target = data.cuda(), target.long().cuda()

        data, target = Variable(data), Variable(target[:,0])

        # transform the whole batch
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
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        for p in model.parameters():
            p.data.add_(-opt.learning_rate, p.grad.data)

        # optimize / backprop
        optimizer.step()

        # get loss and accuracy
        train_loss += loss.data[0]
        train_total += target.size()[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # logging
        if batch_idx % 100 == 0:
            train_loss /= train_total
            accuracy = correct / train_total

            saver.log_string(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t Accuracy: {:.2f}%').format(\
                epoch, batch_idx * len(data), len(train_loader.dataset),\
                100. * batch_idx / len(train_loader), train_loss, accuracy*100.))

            saver.update_training_info(True, train_loss, accuracy, 0, None, epoch*len(train_loader.dataset)+batch_idx*data.size()[0])

def test(model, criterion, saver, test_loader, epoch):
    test_loss = 0.0
    correct = 0.0
    correct_class = [0 for _ in range(40)]
    total_class = [0 for _ in range(40)]
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

        for idx, label in enumerate(target.data.cpu().numpy()):
            total_class[label] += 1
            predicted = pred.cpu().numpy()[idx]
            if predicted == label:
                correct_class[label] += 1

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # logging
    test_loss /= len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)

    avg_class_acc = np.mean(np.array(correct_class)/ np.array(total_class))


    saver.log_string(('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} {:.2f}%, Average Class Accuracy: {:.2f}%\n')
          .format(test_loss, correct, len(test_loader.dataset),
                  100.*epoch_acc, avg_class_acc*100.))

    saver.update_training_info(False, test_loss, epoch_acc, avg_class_acc, model.state_dict(), epoch*len(test_loader.dataset))

    return test_loss
    
def main():
    opt = parse_arguement()

    saver = utils.Saver(opt)

    # randomize seed
    opt.manualSeed = random.randint(1, 10000) # fix seed
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    # load data
    root = "data/modelnet40_ply_hdf5_2048/"
    use_cuda = torch.cuda.is_available()

    transforms_list = []
    random_permute = utils.Random_permute(opt.num_points, delta=opt.distance)
    # load transformations
    if opt.random_input:
        transforms_list.append(random_permute)
        
    # Load dataset / data loader
    train_dataset = data.ModelNetDataset(root,
                            train=True,
                            sort=opt.sort,
                            transform=transforms.Compose(transforms_list),
                            distance=opt.distance)
    train_loader = DataLoader(train_dataset,
                            batch_size=opt.batchSize,
                            shuffle=True,
                            num_workers=opt.workers)

    test_dataset = data.ModelNetDataset(root,
                            train=False,
                            sort=opt.sort,
                            distance=opt.distance)
    test_loader = DataLoader(test_dataset,
                            batch_size=opt.batchSize,
                            shuffle=False,
                            num_workers=opt.workers)

    # define model
    ndim = 6 if opt.distance else 3
    if opt.model == 'lstm':
        model = LSTM(input_dim=ndim, maxout=opt.elem_max)
    elif opt.model == 'lstm_mlp':
        model = LSTM_mlp(input_dim=ndim, maxout=opt.elem_max)
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
        model = model.cuda()#nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    best_model_wts = model.state_dict()

    early_stopping = utils.Early_stopping(opt.early_stopping, patience=15)

    saver.log_parameters(model.parameters())

    for epoch in range(opt.nepoch):
        adjust_learning_rate(opt.learning_rate, optimizer, epoch, saver)

        train(model, optimizer, criterion, saver, train_loader, epoch, opt)

        test_loss = test(model, criterion, saver, test_loader, epoch)

        early_stopping.update(test_loss)
        if early_stopping.stop():
            break

    saver.save_result()

if __name__ == '__main__':
    main()