//File name: scan_formatter.h
//Author: Leo Stanislas
//Date created: 2018/06/20
//Date last modified: 2018/06/29

#ifndef SCAN_FORMATTER_H
#define SCAN_FORMATTER_H

#include <smoke_detection/formatter.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

class ScanFormatter: public Formatter {
public:

    ScanFormatter();

    void callBackVelo(const sensor_msgs::PointCloud2::ConstPtr &scan_strongest,
                      const sensor_msgs::PointCloud2::ConstPtr &scan_last) {
        if (!new_lidar_scan_) {
            pcl::fromROSMsg(*scan_strongest, l_scan_strongest_); // Update last velodyne scan with new scan;
            pcl::fromROSMsg(*scan_last, l_scan_last_); // Update last velodyne scan with new scan;
            new_lidar_scan_ = true;
        }
    }

    void processLoop(){
        if (new_lidar_scan_) {
            processLidarScan();
          new_lidar_scan_ =false;
        }
    }

private:

    message_filters::Subscriber<sensor_msgs::PointCloud2> *v_strongest_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *v_last_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> *v_sync_;

    bool new_lidar_scan_;

};

#endif // SCAN_FORMATTER_H
