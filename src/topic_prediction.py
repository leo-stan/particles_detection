#!/usr/bin/env python

"""
    File name: topic_prediction.py
    Author: Leo Stanislas
    Date created: 2018/06/20
    Date last modified: 2018/06/29
    Python Version: 2.7
"""

from sklearn.externals import joblib
import rospy
import rospkg
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import numpy as np
from LidarDatasetHC import LidarDatasetHC
import argparse
from config import cfg
import time

class TopicPredictor(object):

    def __init__(self, model_file, topic_sub='voxel_map', topic_pub='voxel_map_prediction'):
        self.new_pcl = None
        rospy.Subscriber(topic_sub, Pc2, self.callbackLidarPcl)
        self.model, _, _, features, mu, sigma = joblib.load(
            model_file)
        self.pub = rospy.Publisher(topic_pub, Pc2, queue_size=10)
        self.predset = LidarDatasetHC(features=features, mu=mu, sigma=sigma)
        self.fields = []
        self.computeFields()
        self.average_time = 0
        self.timing_count = 0
        self.timing_count_max = cfg.TIMING_COUNT_MAX


    def callbackLidarPcl(self, pcl):
        if not self.new_pcl:
            self.new_pcl = pcl

    def computeFields(self):
        p_x = Pf('x', 0, 7, 1)
        p_y = Pf('y', 4, 7, 1)
        p_z = Pf('z', 8, 7, 1)
        self.fields = [p_x, p_y, p_z]

        if 'int_mean' in self.predset.features:
            p_int_mean = Pf('int_mean', 12, 7, 1)
            self.fields.append(p_int_mean)

        if 'int_var' in self.predset.features:
            p_int_var = Pf('int_var', 16, 7, 1)
            self.fields.append(p_int_var)

        if 'echo' in self.predset.features:
            p_echo = Pf('echo', 20, 2, 1)
            self.fields.append(p_echo)

        if 'roughness' in self.predset.features:
            p_roughness = Pf('roughness', 21, 7, 1)
            self.fields.append(p_roughness)

        if 'slope' in self.predset.features:
            p_slope = Pf('slope', 25, 7, 1)
            self.fields.append(p_slope)

        p_label = Pf('label', 29, 2, 1)
        self.fields.append(p_label)

    def predictScan(self):
        if self.new_pcl:
            pcl = list(p_c2.read_points(self.new_pcl, field_names=['x', 'y', 'z', 'intensity', 'echo'],
                                        skip_nans=True))
            pcl = np.asarray(pcl)

            if len(pcl) > 0:
                X_pred, coordinate_buffer = self.predset.get_pred_scan(pcl)

                t_start = time.time()
                pred = self.model.predict(X_pred)
                if self.timing_count < self.timing_count_max :
                    self.average_time += (time.time() - t_start)
                    self.timing_count += 1
                    print('Prediction time No %d : %f' % (self.timing_count, (time.time() - t_start)))
                else:
                    print('Average prediction time: %f' % (self.average_time/self.timing_count_max))

                header = self.new_pcl.header
                header.stamp = rospy.Time.now()
                header.frame_id = 'voxel_map'

                voxel_pos = coordinate_buffer * np.array((cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE))
                pred_points = np.concatenate((voxel_pos, X_pred, np.reshape(pred, (-1, 1))), axis=1)

                # pred_points = np.concatenate((voxel_pos[:inputs_len, :], np.reshape(output, (-1, 1))), axis=1)
                pred_pcl = p_c2.create_cloud(header, self.fields, pred_points)
                self.pub.publish(pred_pcl)
            self.new_pcl = None


if __name__ == '__main__':

    rospy.init_node('predict_pcl', anonymous=True)

    parser = argparse.ArgumentParser(description='topic_prediction')

    parser.add_argument('__name', type=str, nargs='?',
                        help='launch file args')
    parser.add_argument('__log', type=str, nargs='?', default='',
                        help='launch file args')
    parser.add_argument('-m', '--model', type=str, nargs='?', default='SVM.pkl',
                        help='model_file.pkl')
    parser.add_argument('--topic_sub', type=str, nargs='?', default='lidar_points_filtered',
                        help='input voxel map topic name')
    parser.add_argument('--topic_pub', type=str, nargs='?', default='HC_prediction',
                        help='predicted voxel map topic name')
    args = parser.parse_args()

    rospack = rospkg.RosPack()
    model = rospack.get_path('smoke_detection') + '/model/saved_models/' + args.model

    tp = TopicPredictor(model, args.topic_sub, args.topic_pub)

    print('Publishing prediction with model "%s" on topic /%s' % (args.model, args.topic_pub))
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        tp.predictScan()
        rate.sleep()
