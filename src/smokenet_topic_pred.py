#!/usr/bin/env python

"""
    File name: topic_prediction.py
    Author: Leo Stanislas
    Date created: 2018/06/20
    Date last modified: 2018/06/29
    Python Version: 2.7
"""

import rospy
import rospkg
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import numpy as np
import argparse
import torch
import time

from LidarDataset import LidarDataset
from config import cfg


class SmokeNetTopicPredictor(object):
    def __init__(self, model_file, topic_sub='smokenet_prediction', topic_pub='smokenet_prediction'):
        self.new_pcl = None
        rospy.Subscriber(topic_sub, Pc2, self.__callbackLidarPcl)  # Declare topic subscriber and assign callback function
        _, self.net, features, mu, sigma, transform = torch.load(model_file)
        self.pub = rospy.Publisher(topic_pub, Pc2, queue_size=10)  # Declare publisher
        self.pub_seg = rospy.Publisher('smokenet_prediction_seg', Pc2, queue_size=10)
        self.__computeFields()
        self.predset = LidarDataset(features=features, mu=mu, sigma=sigma, transform=transform)
        self.net.eval()
        self.average_time = 0
        self.timing_count = 0
        self.timing_count_max = cfg.TIMING_COUNT_MAX
        self.disp_raw_scan = True

        use_gpu = True

        if use_gpu and torch.cuda.is_available():
            # Check what hardware is available
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print('Device used: ' + self.device.type)
        self.net.to(self.device)

    # Private function that is called everytime we receive a Lidar point cloud and load it in self.new_pcl
    def __callbackLidarPcl(self, pcl):
        if not self.new_pcl:
            self.new_pcl = pcl

    # Private function, This function compute the different fields that go into each point of the point cloud Pf(
    # 'field_name', position, data type,1)
    def __computeFields(self):
        p_x = Pf('x', 0, 7, 1)
        p_y = Pf('y', 4, 7, 1)
        p_z = Pf('z', 8, 7, 1)
        self.fields = [p_x, p_y, p_z]

        p_label = Pf('label', 12, 2, 1)
        self.fields.append(p_label)

    # This function is called 10 times per second (in the main function below), checks if there is a new pointcloud received, if there is, process the pointcloud and published a new one
    def predictScan(self):
        if self.new_pcl:
            pcl = list(p_c2.read_points(self.new_pcl, field_names=['x', 'y', 'z', 'intensity', 'echo'],
                                        skip_nans=True))
            pcl = np.asarray(pcl)
            # Check that there are points in the scan
            if len(pcl) > 0:
                feature_buffer, coordinate_buffer = self.predset.get_pred_scan(pcl)
                inputs = feature_buffer.to(self.device)

                with torch.no_grad():
                    t_start = time.time()
                    output = self.net(inputs)
                if self.timing_count < self.timing_count_max:
                    self.average_time += (time.time() - t_start)
                    self.timing_count += 1
                    print('Prediction time No %d : %f' % (self.timing_count, (time.time() - t_start)))
                else:
                    print('Average prediction time: %f' % (self.average_time / self.timing_count_max))

                # batch_size = feature_buffer.shape[0]
                # # batch_size = 64
                # inputs_len = int(inputs.shape[0] / batch_size) * batch_size
                # inputs = np.reshape(inputs[:inputs_len, :, :],
                #                     (-1, batch_size, cfg.VOXEL_POINT_COUNT, feature_buffer.shape[2]))
                # inputs.to(self.device)
                # output = []
                # with torch.no_grad():
                #
                #     for i in range(0, inputs.shape[0]):
                #         output.append(self.net(inputs[i, :, :, :]))
                #
                output = output.argmax(dim=1)

                # # pred = self.model.predict(self.scaler.transform(X))
                #
                # pred_points = np.concatenate((pcl, X, np.reshape(pred, (pred.size, 1))), axis=1)

                header = self.new_pcl.header
                header.stamp = rospy.Time.now()
                header.frame_id = 'voxel_map'

                if self.disp_raw_scan:
                    pred_points = segment_scan(self.new_pcl, output)
                else:
                    voxel_pos = coordinate_buffer * np.array((cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE))
                    pred_points = np.concatenate((voxel_pos, np.reshape(output, (-1, 1))), axis=1)
                pred_pcl = p_c2.create_cloud(header, self.fields, pred_points)
                self.pub.publish(pred_pcl)
                pred_pcl_seg = p_c2.create_cloud(header, self.fields, pred_points[pred_points[:,3] == 0,:])
                self.pub_seg.publish(pred_pcl_seg)
            self.new_pcl = None


def segment_scan(raw_lidar_pcl, output):
    raw_lidar_pcl = list(p_c2.read_points(raw_lidar_pcl, field_names=['x', 'y', 'z', 'intensity', 'echo'],
                                          skip_nans=True))
    raw_lidar_pcl = np.asarray(raw_lidar_pcl)

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

    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(len(coordinate_buffer)):
        index_buffer[tuple(coordinate_buffer[i])] = i

    raw_pcl_label = np.zeros((len(voxel_index),1))
    for i in range(len(voxel_index)):
        raw_pcl_label[i] = output[index_buffer[tuple(voxel_index[i, :])]]
    pred_raw_lidar_pcl = np.append(shifted_coord[bound_box],raw_pcl_label,axis=1)

    return pred_raw_lidar_pcl


if __name__ == '__main__':

    rospy.init_node('predict_pcl', anonymous=True)

    parser = argparse.ArgumentParser(description='topic_prediction')

    parser.add_argument('__name', type=str, nargs='?',
                        help='launch file args')
    parser.add_argument('__log', type=str, nargs='?', default='',
                        help='launch file args')
    parser.add_argument('-m', '--model', type=str, nargs='?', default='int_relpos.pt',
                        help='model_file.pkl')
    parser.add_argument('--topic_sub', type=str, nargs='?', default='lidar_points_filtered',
                        help='input voxel map topic name')
    parser.add_argument('--topic_pub', type=str, nargs='?', default='smokenet_prediction',
                        help='predicted voxel map topic name')
    args = parser.parse_args()

    rospack = rospkg.RosPack()
    model = rospack.get_path('smoke_detection') + '/model/saved_models/' + args.model

    de = SmokeNetTopicPredictor(model, args.topic_sub, args.topic_pub)

    print('Publishing prediction with model "%s" on topic /%s' % (args.model, args.topic_pub))
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        de.predictScan()
        rate.sleep()
