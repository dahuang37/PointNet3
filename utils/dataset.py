import os
import sys
import h5py
import json

TRAIN_FILE_LIST = "data/modelnet40_ply_hdf5_2048/train_files.txt"
TRAIN_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/train_files_id.txt"
TEST_FILE_LIST  = "data/modelnet40_ply_hdf5_2048/test_files.txt"
TEST_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/test_files_id.txt"


def load_json(json_filename):
    with open(json_filename) as file:
        data = json.load(file)
    print(data)

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def getDataFiles(file_list):
    return [line.rstrip() for line in open(file_list)]

def getTrainingData(index):
    train_file_list = getDataFiles(TRAIN_FILE_LIST)
    return loadDataFile(train_file_list[index])

def getTestingData(index):
    test_files_list = getDataFiles(TEST_FILE_LIST)
    return loadDataFile(test_files_list[index])
