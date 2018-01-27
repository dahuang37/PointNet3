import torch
import numpy as np


class Random_permute(object):
    def __init__(self, num_points=2048):
        self.num_points = num_points
    def __call__(self, sample):
# #def permute_transoform(data, num_points=2048):
#         print(sample.shape)
        permutations = torch.randperm(self.num_points)
        data_cat = sample[permutations]
        return data_cat.astype(sample.dtype)


class Rotate_point_cloud(object):
    def __call__(self, sample):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original batch of point clouds
            Return:
              Nx3 array, rotated batch of point clouds
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        
        ret  = np.dot(sample.reshape((-1, 3)), rotation_matrix)
        return ret.astype(sample.dtype)
        # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        # for k in range(batch_data.shape[0]):
        #     rotation_angle = np.random.uniform() * 2 * np.pi
        #     cosval = np.cos(rotation_angle)
        #     sinval = np.sin(rotation_angle)
        #     rotation_matrix = np.array([[cosval, 0, sinval],
        #                                 [0, 1, 0],
        #                                 [-sinval, 0, cosval]])
        #     shape_pc = batch_data[k, ...]
        #     rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        # return rotated_data


class Jitter_point_cloud(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip    
    def __call__(self, sample):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        N, C = sample.shape
        assert(self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
        jittered_data += sample
        return jittered_data.astype(sample.dtype)