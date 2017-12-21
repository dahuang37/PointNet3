from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

import argparse
import h5py
import json

TRAIN_FILE_LIST = "data/modelnet40_ply_hdf5_2048/train_files.txt"
TRAIN_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/train_files_id.txt"
TEST_FILE_LIST  = "data/modelnet40_ply_hdf5_2048/test_files.txt"
TEST_FILE_ID_LIST = "data/modelnet40_ply_hdf5_2048/test_files_id.txt"
SHAPE_LIST = "data/modelnet40_ply_hdf5_2048/shape_names.txt"

class ModelNetDataset(Dataset):
    def __init__(self, root, train=True, npoints=2048, mesh=False):
        """
        Args:
            root (string): Directory of data, default = data/modelnet40_ply_hdf5_2048
            train (boolean): return Training data if true, Testing data o.w.
            npoints (int): 512, 1024, 2048; default: 2048

        Returns:
            data (n, npoints, 3): ndarray image arrays
            labels (n, 1): ndarray, label of image
            ###label_names (n,1): str ndarray label names 
            ###mesh_paths (str): director of its corresponding mesh
        """
        
        def load_mesh_file(filename, train=True):
            """
            parse the list of mesh directories to match '.off' file in data/ModelNet40 folder
            """
            with open(filename) as file:
                data = json.load(file)
            # original: eg, sofa/sofa_0037.ply, output: ['sofa', 'sofa_0037]
            a = [data[i].split('.')[0].split('/') for i in range(len(data))]
            # output: "data/ModelNet40/sofa/train/sofa_0037.off"
            if train:
                folder = 'train'
            f = [os.path.join("data/ModelNet40",a[i][0], folder,a[i][1]+'.off')  for i in range(len(a))]
            return f
        
        train_txt = os.path.join(root, "train_files.txt")
        mesh_train_txt = os.path.join(root, "train_files_id.txt")
        test_txt = os.path.join(root, "test_files.txt")
        mesh_test_txt = os.path.join(root, "test_files_id.txt")
        label_names_txt = os.path.join(root, "shape_names.txt")
        
        label_names = [line.rstrip() for line in open(label_names_txt)]
        if train:
            h5_list = [line.rstrip() for line in open(train_txt)]
            mesh_list = [line.rstrip() for line in open(mesh_train_txt)]
        else:
            h5_list = [line.rstrip() for line in open(test_txt)]
            mesh_list = [line.rstrip() for line in open(mesh_test_txt)]
        
        h5_file = h5py.File(h5_list[0])
        self.npoints = npoints
        self.mesh = mesh
        self.data = h5_file['data'][:,0:npoints,:]
        self.labels = h5_file['label']
        self.length = len(self.data)
        self.mesh_paths = load_mesh_file(mesh_list[0])
        print(len(self.mesh_paths))
        
        for i in range(1, len(h5_list)):
            h5_file = h5py.File(h5_list[i])
            datum = h5_file['data'][:,0:npoints,:]
            label = h5_file['label']
            self.data = np.concatenate((self.data, datum), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)
            self.length += len(datum)
            mesh_path = load_mesh_file(mesh_list[i])
            self.mesh_paths += mesh_path
            
    def __getitem__(self, index):
        if self.mesh:
            return self.data[index], self.labels[index], self.mesh_paths[index]
        else:
            return self.data[index], self.labels[index]
    
    def __len__(self):
        return self.length