#!/usr/bin/env python

""""
    File name: LidarDataset.py
    Author: Leo Stanislas
    Date created: 2018/08/14
    Python Version: 2.7
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os, os.path
import glob
from config import cfg
import sys


class LidarDataset(Dataset):
    """Lidar dataset."""

    def __init__(self, features, root_dir=None, train=True, test=False, val=False, smoke=True, dust=True, shuffle=False,
                 mu=None, sigma=None, transform=None):
        """
        Args:
            root_dir (string): Directory with training data:
            |---root_dir
                |--- label
                |--- lidar
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.labels = []
        if len(features) == 0:
            sys.exit('LidarDataset: nb_features = 0, need at least one feature')
        self.features = np.asarray(features)
        self.pos = 'pos' in self.features
        self.vox_pos = 'vox_pos' in self.features
        self.intensity = 'intensity' in self.features
        self.echo = 'echo' in self.features
        self.rel_pos = 'rel_pos' in self.features

        self.nb_features = 0
        if self.pos or self.vox_pos:
            self.nb_features += 3
        if self.intensity:
            self.nb_features += 1
        if self.echo:
            self.nb_features += 1
        if self.rel_pos:
            self.nb_features += 3

        self.root_dir = root_dir
        self.transform = transform
        self.smoke = smoke
        self.dust = dust
        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(self.shuffle)
        # Scaling parameters
        self.mu = mu
        self.sigma = sigma
        if (self.mu is not None or self.sigma is not None) and len(mu) != len(sigma):
            sys.exit('Mu and Sigma need to be of same dimension')

        # Prepare Dataset
        self.smoke_data = []
        self.non_smoke_data = []
        self.smoke_labels = []
        self.non_smoke_labels = []

        # If root directory provided, build dataset out of that
        if root_dir is not None:
            self.f_lidar = []
            self.f_label = []

            if train:
                self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, 'training/lidar/*.bin'))
                self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, 'training/label/*.txt'))
            if test:
                self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, 'testing/lidar/*.bin'))
                self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, 'testing/label/*.txt'))
            if val:
                self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, 'validation/lidar/*.bin'))
                self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, 'validation/label/*.txt'))

            self.f_lidar.sort()
            self.f_label.sort()

            for i in range(len(self.f_lidar)):
                raw_lidar_pcl = np.fromfile(self.f_lidar[i], dtype=np.float32).reshape((-1, 5))
                if raw_lidar_pcl.size != 0:
                    label = int([line for line in open(self.f_label[i], 'r').readlines()][0])
                    if (label == 1 and not smoke) or (label == 2 and not dust):
                        break  # If unrequested label, skip
                    feature_buffer, _ = self.extract_scan(raw_lidar_pcl)
                    if label == 0:
                        self.non_smoke_data.append(feature_buffer)
                        self.non_smoke_labels.append(label * np.ones(shape=(feature_buffer.shape[0], 1), dtype=np.int64))
                    else:
                        label = 1
                        self.smoke_data.append(feature_buffer)
                        self.smoke_labels.append(label * np.ones(shape=(feature_buffer.shape[0], 1), dtype=np.int64))
            self.non_smoke_data = np.concatenate(self.non_smoke_data)
            self.smoke_data = np.concatenate(self.smoke_data)
            self.non_smoke_labels = np.concatenate(self.non_smoke_labels)
            self.smoke_labels = np.concatenate(self.smoke_labels)

            self.post_process_scan()

    def extract_scan(self, raw_lidar_pcl):
        if self.shuffle:
            np.random.shuffle(raw_lidar_pcl)

        # Process raw scan
        # Apply -90deg rotation on z axis to go from robot to map
        raw_lidar_pcl[:, [0, 1]] = raw_lidar_pcl[:, [1, 0]]  # Swap x, y axis
        raw_lidar_pcl[:, :3] = raw_lidar_pcl[:, :3] * np.array([-1, 1, 1], dtype=np.int8)

        # Translation from map to robot_footprint
        husky_footprint_coord = np.array([cfg.MAP_TO_FOOTPRINT_X, cfg.MAP_TO_FOOTPRINT_Y, cfg.MAP_TO_FOOTPRINT_Z],
                                         dtype=np.float32)

        # Lidar points in map coordinate
        shifted_coord = raw_lidar_pcl[:, :3] + husky_footprint_coord

        voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
        grid_size = np.array([(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE, (cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE,
                              (cfg.Z_MAX - cfg.Z_MIN) / cfg.VOXEL_Z_SIZE], dtype=np.int64)

        voxel_index = np.floor(shifted_coord / voxel_size).astype(np.int)

        bound_x = np.logical_and(
            voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])
        bound_y = np.logical_and(
            voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
        bound_z = np.logical_and(
            voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])

        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

        # Raw scan within bounds
        raw_lidar_pcl = raw_lidar_pcl[bound_box]
        voxel_index = voxel_index[bound_box]

        # [K, 3] coordinate buffer as described in the paper
        coordinate_buffer = np.unique(voxel_index, axis=0)

        # Number of voxels in scan
        K = len(coordinate_buffer)
        # Max number of lidar points in each voxel
        T = cfg.VOXEL_POINT_COUNT

        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)

        # [K, T, 8] feature buffer as described in the paper
        feature_buffer = np.zeros(shape=(K, T, 8), dtype=np.float32)

        # build a reverse index for coordinate buffer
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i

        for voxel, point in zip(voxel_index, raw_lidar_pcl):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < T:
                feature_buffer[index, number, :5] = point
                number_buffer[index] += 1
        # Added this step to only predict voxel-wise features for cells with points
        for i in range(K):
            feature_buffer[i, :number_buffer[i], -3:] = feature_buffer[i, :number_buffer[i], :3] - \
                                                        feature_buffer[i, :number_buffer[i], :3].sum(axis=0,
                                                                                                     keepdims=True) / \
                                                        number_buffer[i]

        # Pick and choose features here
        selected_buffer = np.array([], dtype=np.float32).reshape(feature_buffer.shape[0], feature_buffer.shape[1], 0)

        if self.pos:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 0:3]), axis=2)
        if self.intensity:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 3:4]), axis=2)
        if self.echo:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
        if self.rel_pos:
            selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 5:8]), axis=2)

        return selected_buffer, coordinate_buffer

    def post_process_scan(self):
        # Shuffle prior to concatenation to avoid taking the same samples everytime
        if self.shuffle:
            p = np.random.permutation(len(self.non_smoke_data))
            self.non_smoke_data = self.non_smoke_data[p]
            self.non_smoke_labels = self.non_smoke_labels[p]
            p = np.random.permutation(len(self.smoke_data))
            self.smoke_data = self.smoke_data[p]
            self.smoke_labels = self.smoke_labels[p]

        # Adjust number of each label to be equal
        if self.non_smoke_data.shape[0] > self.smoke_data.shape[0]:
            self.non_smoke_data = self.non_smoke_data[:self.smoke_data.shape[0], :, :]
            self.non_smoke_labels = self.non_smoke_labels[:self.smoke_labels.shape[0]]
        else:
            self.smoke_data = self.smoke_data[:self.non_smoke_data.shape[0], :, :]
            self.smoke_labels = self.smoke_labels[:self.non_smoke_labels.shape[0]]

        self.data = np.concatenate((self.smoke_data, self.non_smoke_data), axis=0)
        self.labels = np.concatenate((self.smoke_labels, self.non_smoke_labels), axis=0)
        self.labels = self.labels.reshape((self.labels.shape[0]))
        # Need to re shuffle post concatenation
        if self.shuffle:
            p = np.random.permutation(len(self.data))
            self.data = self.data[p]
            self.labels = self.labels[p]

        self.normalise_data()

    def get_pred_scan(self, raw_lidar_pcl):
        self.data, coord_buffer = self.extract_scan(raw_lidar_pcl)
        self.normalise_data()
        if self.transform:
            self.data = self.transform(self.data)
        return self.data, coord_buffer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        data, labels = self.data[idx], self.labels[idx]

        if self.transform:
            data = self.transform(data)

        sample = {'inputs': data, 'labels': labels}

        return sample

    def get_scale(self):
        data_tmp = np.reshape(self.data, (-1, self.data.shape[2]))
        return np.mean(data_tmp, axis=0), np.std(data_tmp, axis=0)

    def update_transform(self, transform):
        self.transform = transform

    def normalise_data(self):
        # Data normalisation
        if self.mu is None or self.sigma is None:
            print('No scaling values provided, calulating scaling values for normalisation...')
            data_tmp = np.reshape(self.data, (-1, self.data.shape[2]))
            data_tmp = data_tmp[~np.all(data_tmp == 0, axis=1)]

            self.mu = np.mean(data_tmp, axis=0)
            self.sigma = np.std(data_tmp, axis=0)

        if self.nb_features != len(self.mu) or self.nb_features != len(self.sigma):
            sys.exit('Number of features different than mu or sigma')

        idx = np.arange(0, cfg.VOXEL_POINT_COUNT)
        for i in range(0, self.data.shape[0]):
            self.data[i, idx[~np.all(self.data[i, :, :] == 0, axis=1)], :] = (self.data[i, idx[~np.all(
                self.data[i, :, :] == 0, axis=1)], :] - self.mu) / self.sigma


class ToTensor(object):
    """Torch transform: Convert ndarrays in sample to Tensors."""

    def __call__(self, inputs):
        return torch.from_numpy(inputs).type(torch.FloatTensor)


class Normalise(object):
    """Torch transform: normalise data."""

    def __init__(self, mu, sigma):
        if len(mu) == len(sigma):
            self.mu = mu
            self.sigma = sigma
        else:
            sys.exit('Mu and Sigma need to be of same dimension')

    def __call__(self, inputs):

        if inputs.shape[1] == len(self.mu):
            return (inputs - self.mu) / self.sigma
        else:
            sys.exit('Number of features different than mu and sigma')
