import numpy as np
import warnings
import h5py
import os
import random
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def load_data_simple(dir,classification = False):
    data_train0, label_train0, Seglabel_train0  = load_h5(os.path.join(dir, 'ply_data_train0.h5'))

    data_test0, label_test0, Seglabel_test0 = load_h5(os.path.join(dir, 'ply_data_test0.h5'))


    return data_train0, label_train0, data_test0, label_test0


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
    return jittered_data


def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(os.path.join(dir, 'ply_data_train0.h5'))
    data_train1, label_train1,Seglabel_train1 = load_h5(os.path.join(dir, 'ply_data_train1.h5'))
    data_train2, label_train2,Seglabel_train2 = load_h5(os.path.join(dir, 'ply_data_train2.h5'))
    data_train3, label_train3,Seglabel_train3 = load_h5(os.path.join(dir, 'ply_data_train3.h5'))
    data_train4, label_train4,Seglabel_train4 = load_h5(os.path.join(dir, 'ply_data_train4.h5'))
    data_test0, label_test0,Seglabel_test0 = load_h5(os.path.join(dir, 'ply_data_test0.h5'))
    data_test1, label_test1,Seglabel_test1 = load_h5(os.path.join(dir, 'ply_data_test1.h5'))
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, maxpoint = 1024,rotation = None):
        self.data = data
        self.labels = labels
        self.rotation = rotation
        self.maxpoint = maxpoint

    def __len__(self):
        return len(self.data)

    def jitter_point_cloud(self, points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        N, C = points.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += points
        return jittered_data

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix).astype(np.float32)

        return rotated_data

    def random_sample(self, xyz, npoint):
        N, C = xyz.shape
        rs = np.array(random.sample(range(0, N), npoint))
        xyz = xyz[rs]
        return xyz

    def __getitem__(self, index):
        '''

        :param index:
        :return: data [B,C,S]
                labels [B,1]
        '''
        norm_para = 1
        if self.rotation is not None: # for train
            if self.rotation[0] != self.rotation[1]:
                pointcloud = self.data[index]
                angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
                pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)

                cat_data = np.concatenate([pointcloud, pointcloud / norm_para], axis=1)

            else:
                points = self.data[index]
                # points = self.random_sample(points, 1024)
                points = points[0:self.maxpoint,:]
                # points = self.jitter_point_cloud(points).astype(np.float32)

                # cat_data = np.concatenate([points, points / norm_para], axis=1)
                cat_data = np.concatenate([points, points / points], axis=1)
            return cat_data, self.labels[index]
        else:
            # for test
            points = self.data[index]
            points = points[0:self.maxpoint, :]
            # cat_data = np.concatenate([points, points / norm_para], axis=1)
            cat_data = np.concatenate([points, points / points], axis=1)

            # cat_data = np.concatenate([self.data[index], self.data[index] / norm_para], axis=1)

            return cat_data, self.labels[index]

if __name__ == '__main__':
    root = r'C:\Users\PC\Desktop\PointNet_pytroch_77star\data\modelnet\modelnet40'

    # file_list = os.walk(root)
    # for dirpath, _, filenames in file_list:
    #     for f_name in filenames:
    #     # if f_name[-3:] == '.h5':
    #         path = os.path.join(dirpath,f_name)
    #         f = load_h5(path)
    #         print(f[0])

    import torch
    train_data, train_label, test_data, test_label = load_data(root, classification=True)
    trainDataset = ModelNetDataLoader(train_data, train_label, rotation=None)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(trainDataLoader):
        points,label = batch
        print(points.shape)

