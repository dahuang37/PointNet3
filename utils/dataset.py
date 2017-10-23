from __future__ import print_function
#import torch.utils.data as data
from PIL import Image
import os
import sys
import errno
#import torch
import numpy as np
#import progressbar
#import torchvision.transforms as transforms
import argparse
import h5py
import json


# class PartDataSet(data.dataset):
#     def __init__(self, root, num_points=2400, class_choice=None, train=True):
#         self.root = root
#         self.num_points = num_points
#         self.class_choice = class_choice
#         self.train = train
#         self.filepath = os.path.join(self.root, 'train_files.txt')
#     def __len__(self):
#         return len(self.datapath)

TRAIN_FILE_LIST = "data/modelnet40_ply_hdf5_2048/train_files.txt"
TRAIN_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/train_files_id.txt"
TEST_FILE_LIST  = "data/modelnet40_ply_hdf5_2048/test_files.txt"
TEST_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/test_files_id.txt"
SHAPE_LIST = "data/modelnet40_ply_hdf5_2048/shape_names.txt"


def load_json(json_filename):
    with open(json_filename) as file:
        data = json.load(file)
    return data

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def getDataFiles(file_list):
    return [line.rstrip() for line in open(file_list)]

def getShapeNames():
    return getDataFiles(SHAPE_LIST)

def get3DList():
    train_id_list = getDataFiles(TRAIN_FILE_ID_LIST)
    files = load_json(train_id_list[0])
    for i in range(1, len(train_id_list)):
        f = load_json(train_id_list[i])
        files = np.concatenate((files, f), axis=0)
    return files

def getTrainingData(index):
    train_file_list = getDataFiles(TRAIN_FILE_LIST)
    return loadDataFile(train_file_list[index])

def getAllTrainingData():
    train_file_list = getDataFiles(TRAIN_FILE_LIST)
    data, label = loadDataFile(train_file_list[0])
    for i in range(1,len(train_file_list)):
        d, l = loadDataFile(train_file_list[i])
        data = np.concatenate((data, d), axis=0)
        label = np.concatenate((label,l), axis=0)
    return (data,label)

def getTestingData(index):
    test_files_list = getDataFiles(TEST_FILE_LIST)
    return loadDataFile(test_files_list[index])
