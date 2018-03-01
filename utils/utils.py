import torch
import numpy as np


class Early_stopping(object):
    '''
    monitor on val loss
    if stop improving from the past 10 epoch, then stop
    '''
    def __init__(self, in_use, patience=15):
        self.in_use = in_use
        self.patience = patience
        self.wait = 0
        self.current_best = np.inf
        self.stopping = False

    def update(self, test_loss):
        if not self.in_use:
            return
        if self.current_best > test_loss:
            self.current_best = test_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopping = True

    def stop(self):
        return self.stopping

class Random_permute(object):
    def __init__(self, num_points=2048, delta=False):
        self.num_points = num_points
        self.delta = delta
    def __call__(self, sample, delta=False):
        permutations = torch.randperm(self.num_points)
        data_cat = sample[permutations]

        if self.delta:
            new_data = np.zeros((sample.shape[0],sample.shape[1], 6), dtype=sample.dtype)
            new_data[:,:,:3] = data_cat
            for j in range(sample.shape[1]-1):
               new_data[:,j,3] = data_cat[:,j+1,0] - data_cat[:,j,0]
               new_data[:,j,4] = data_cat[:,j+1,1] - data_cat[:,j,1]
               new_data[:,j,5] = data_cat[:,j+1,2] - data_cat[:,j,2]
            return new_data

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


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * np.pi * 1.0
        # print(rotation_angle)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    # subtracting by mean
    centroid = np.mean(rotated_data, axis=0)
    rotated_data = rotated_data - centroid
    # m = np.max(np.sqrt(np.sum(rotated_data**2,axis=1)))
    # rotated_data = rotated_data / m
    
    return rotated_data.astype(batch_data.dtype)


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data.astype(batch_data.dtype)


