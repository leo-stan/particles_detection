#!/usr/bin/env python

""""
    File name: extract_rosbags.py
    Author: Leo Stanislas
    Date created: 2018/09/02
    Python Version: 2.7
"""

import rosbag
import sensor_msgs.point_cloud2 as p_c2
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data_dir = '/home/leo/phd/smoke_detection/src/smoke_detection/data'

    sensors = {
        'lidar_points_filtered': 'lidar',
        'stereo_points_filtered': 'stereo'
    }

    field_names = {
        'lidar': [
            'x',
            'y',
            'z',
            'intensity',
            'echo'],
        'stereo': [
            'x',
            'y',
            'z',
            'r',
            'g',
            'b']
    }

    # List of bag files to open
    training_bags = [
        '/home/leo/phd/smoke_detection/src/smoke_detection/data/rosbags/non_particle.bag',
        '/home/leo/phd/smoke_detection/src/smoke_detection/data/rosbags/bush.bag',
        '/home/leo/phd/smoke_detection/src/smoke_detection/data/rosbags/smoke.bag',
        '/home/leo/phd/smoke_detection/src/smoke_detection/data/rosbags/dust.bag',
    ]

    pc_s = []
    pc_l = []
    labels = []
    lidar = False
    stereo = False

    # Clean data directory
    folders = ['training/lidar',
               'training/stereo',
               'training/label',
               'validation/lidar',
               'validation/stereo',
               'validation/label',
               'testing/lidar',
               'testing/stereo',
               'testing/label',
               ]
    for folder in folders:
        for the_file in os.listdir(os.path.join(data_dir,folder)):
            file_path = os.path.join(data_dir,folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(e)

    # Go through list of bag files
    for bag in training_bags:

        print('Extracting %s...' % bag)
        bag_data = rosbag.Bag(bag)  # load bag from string name
        msgs = bag_data.read_messages()  # read messages in bag

        # Go through each message in the bag
        for topic, msg, t in msgs:

            # Check the topic that the msg came from
            if topic == 'lidar_points_filtered':
                # Load pointcloud with appropriate field name
                pc_l.append(np.asarray(
                    list(p_c2.read_points(msg, field_names=field_names['lidar'], skip_nans=True)),
                    dtype=np.float32))
            elif topic == 'stereo_points_filtered':
                pc_s.append(np.asarray(
                    list(p_c2.read_points(msg, field_names=field_names['stereo'], skip_nans=True)),
                    dtype=np.float32))
            elif topic == 'label':
                labels.append(msg.data)

    if len(pc_l) != 0:
        lidar = True
    if len(pc_s) != 0:
        stereo = True

    if not (lidar or stereo):
        os.error('Need at least lidar or stereo')
        sys.exit()

    if lidar and stereo and len(pc_l) != len(pc_s):
        os.error('Sensor scans have different sizes')
        sys.exit()

    nb_scans = len(labels)

    if lidar and len(pc_l) != nb_scans:
        os.error('Lidar scans and labels are different sizes')
        sys.exit()

    if stereo and len(pc_s) != nb_scans:
        os.error('Stereo scans and labels are different sizes')
        sys.exit()

    # TODO: maybe remove empty scans

    train, test = train_test_split(range(0, nb_scans), test_size=.4, random_state=42)
    val = test[:len(test) / 2]
    test = test[len(test) / 2:]

    for i in train:
        if lidar:
            pc_l[i].tofile(os.path.join(data_dir, 'training/lidar', '{:06d}.bin'.format(i)))
        if stereo:
            pc_s[i].tofile(os.path.join(data_dir, 'training/stereo', '{:06d}.bin'.format(i)))
        label_file = open(os.path.join(data_dir, 'training/label', '{:06d}.txt'.format(i)), 'w')
        label_file.write(str(labels[i]))
        label_file.close()

    for i in test:
        if lidar:
            pc_l[i].tofile(os.path.join(data_dir, 'testing/lidar', '{:06d}.bin'.format(i)))
        if stereo:
            pc_s[i].tofile(os.path.join(data_dir, 'testing/stereo', '{:06d}.bin'.format(i)))
        label_file = open(os.path.join(data_dir, 'testing/label', '{:06d}.txt'.format(i)), 'w')
        label_file.write(str(labels[i]))
        label_file.close()

    for i in val:
        if lidar:
            pc_l[i].tofile(os.path.join(data_dir, 'validation/lidar', '{:06d}.bin'.format(i)))
        if stereo:
            pc_s[i].tofile(os.path.join(data_dir, 'validation/stereo', '{:06d}.bin'.format(i)))
        label_file = open(os.path.join(data_dir, 'validation/label', '{:06d}.txt'.format(i)), 'w')
        label_file.write(str(labels[i]))
        label_file.close()
