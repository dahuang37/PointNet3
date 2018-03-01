from __future__ import print_function, division
import os
import numpy as np
import argparse
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from utils import *


class ModelNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None, npoints=2048, mesh=False, sort=True, distance=False, test_transform=False):
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
        self.root = root
        self.train = train
        self.npoints = npoints
        self.train = train
        self.sort = sort
        self.distance = distance
        self.mesh = mesh

        if self.mesh:
            self.load_mesh_file()

        self.load_data_from_h5()
                
        self.transform = transform
        
        if self.sort:
            self.sort_input()

        if self.distance:
            self.add_delta_input()
            print("Input data size: ")     
            print(self.new_data.shape)
        else:
            print("Input data size: ")     
            print(self.data.shape)


    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]

        # if distance, don't transform / augment data, distance would not match
        if self.distance:
            return self.new_data[index], labels

        if self.transform is not None:
            data = self.transform(data)

        if self.mesh:
            mesh_paths = self.mesh_paths[index]
            return data, labels, mesh_paths

        return data, labels

    def __len__(self):
        return self.length

    def load_mesh_file(self, filename, train=True):
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

    def sort_input(self):
        for num_ex in range(self.data.shape[0]):
            ind = np.lexsort(np.transpose(self.data[i]))
            self.data[i] = self.data[i,ind]

    def add_delta_input(self):
        self.new_data = np.zeros((self.data.shape[0],self.data.shape[1], 6), dtype=np.float32)
        self.new_data[:,:,:3] = self.data
        for j in range(self.data.shape[1]-1):
            self.new_data[:,j,3] = self.data[:,j+1,0] - self.data[:,j,0]
            self.new_data[:,j,4] = self.data[:,j+1,1] - self.data[:,j,1]
            self.new_data[:,j,5] = self.data[:,j+1,2] - self.data[:,j,2]

    def load_data_from_h5(self):
        train_txt = os.path.join(self.root, "train_files.txt")
        test_txt = os.path.join(self.root, "test_files.txt")
        label_names_txt = os.path.join(self.root, "shape_names.txt")

        label_names = [line.rstrip() for line in open(label_names_txt)]
        if self.train:
            h5_list = [line.rstrip() for line in open(train_txt)]
        else:
            h5_list = [line.rstrip() for line in open(test_txt)]

        h5_file = h5py.File(h5_list[0])
        
        self.data = h5_file['data'][:,0:self.npoints,:]
        self.labels = h5_file['label']
        self.length = len(self.data)
        
        for i in range(1, len(h5_list)):
            h5_file = h5py.File(h5_list[i])
            datum = h5_file['data'][:,0:self.npoints,:]
            label = h5_file['label']
            self.data = np.concatenate((self.data, datum), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)
            self.length += len(datum)
            # if self.mesh:
            #     mesh_path = self.load_mesh_file(mesh_list[i])
                # self.mesh_paths += mesh_path

    def add_mesh_paths(self):
        if self.train:
            mesh_train_txt = os.path.join(self.root, "train_files_id.txt")
            mesh_list = [line.rstrip() for line in open(mesh_train_txt)]
        else:
            mesh_test_txt = os.path.join(self.root, "test_files_id.txt")
            mesh_list = [line.rstrip() for line in open(mesh_test_txt)]
        self.mesh_paths = self.load_mesh_file(mesh_list[0])
