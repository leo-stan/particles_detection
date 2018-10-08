//File name: scan_formatter.cpp
//Author: Leo Stanislas
//Date created: 2018/06/20
//Date last modified: 2018/06/29

#include <smoke_detection/scan_formatter.h>


ScanFormatter::ScanFormatter() :
        new_lidar_scan_(false){

    // Subscribe to both echo topics
    v_strongest_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(n_, "/velodyne_points_dual", 5);
    v_last_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(n_, "/velodyne_points", 5);

    // Register both topic time synchronized
    v_sync_ = new message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2>(
            *v_strongest_sub_, *v_last_sub_, 5);
    v_sync_->registerCallback(boost::bind(&ScanFormatter::callBackVelo, this, _1, _2));

}


int main(int argc, char **argv) {
    ros::init(argc, argv, "scan_formatter");
    ros::Time::init();

    ros::Rate r(10); // Update the map at 10 Hz

    ScanFormatter scan_formatter;
    while (ros::ok()) {
        ros::spinOnce();
        scan_formatter.processLoop();
        r.sleep();
    }

    return 0;
}
