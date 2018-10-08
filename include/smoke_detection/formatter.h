//File name: formatter.h
//Author: Leo Stanislas
//Date created: 2018/06/20
//Date last modified: 2018/06/29

#ifndef FORMATTER_H
#define FORMATTER_H

#include <ros/ros.h>
#include <ros/types.h>

#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/rawdata.h>

struct LidarPointFused {
    PCL_ADD_POINT4D;                    // quad-word XYZ
    uint8_t ring;
    uint8_t intensity;
    uint8_t echo;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
}EIGEN_ALIGN16;

// Register pointcloud to be able to publish it
POINT_CLOUD_REGISTER_POINT_STRUCT(LidarPointFused,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (uint8_t, ring, ring)
                                          (uint8_t, intensity, intensity)
                                          (uint8_t, echo, echo))

typedef pcl::PointCloud<LidarPointFused> LidarScanFused;

class Formatter {
public:

    Formatter() :
            scan_no_(0),
            cube_out_min_x_(-std::numeric_limits<float>::max()),
            cube_out_max_x_(std::numeric_limits<float>::max()),
            cube_out_min_y_(-std::numeric_limits<float>::max()),
            cube_out_max_y_(std::numeric_limits<float>::max()),
            cube_out_min_z_(-std::numeric_limits<float>::max()),
            cube_out_max_z_(std::numeric_limits<float>::max()),
            cube_in_min_x_(0),
            cube_in_max_x_(0),
            cube_in_min_y_(0),
            cube_in_max_y_(0),
            cube_in_min_z_(0),
            cube_in_max_z_(0),
            sensor_pos_x_(0),
            sensor_pos_y_(0),
            sensor_pos_z_(1.45) {

        n_.param("cube_in_min_x", cube_in_min_x_, cube_in_min_x_);
        n_.param("cube_in_max_x", cube_in_max_x_, cube_in_max_x_);
        n_.param("cube_in_min_y", cube_in_min_y_, cube_in_min_y_);
        n_.param("cube_in_max_y", cube_in_max_y_, cube_in_max_y_);
        n_.param("cube_in_min_z", cube_in_min_z_, cube_in_min_z_);
        n_.param("cube_in_max_z", cube_in_max_z_, cube_in_max_z_);
        n_.param("cube_out_min_x", cube_out_min_x_, cube_out_min_x_);
        n_.param("cube_out_max_x", cube_out_max_x_, cube_out_max_x_);
        n_.param("cube_out_min_y", cube_out_min_y_, cube_out_min_y_);
        n_.param("cube_out_max_y", cube_out_max_y_, cube_out_max_y_);
        n_.param("cube_out_min_z", cube_out_min_z_, cube_out_min_z_);
        n_.param("cube_out_max_z", cube_out_max_z_, cube_out_max_z_);

        velo_to_footprint_ << 0, 1, 0, sensor_pos_x_,
                -1, 0, 0, sensor_pos_y_,
                0, 0, 1, sensor_pos_z_,
                0, 0, 0, 1;

        pubPcl_ = n_.advertise<LidarScanFused>("/lidar_points_filtered", 10);
    }


    // Check that a point is outside the cube and inside pc boundaries
    const bool checkScanLimits(const velodyne_rawdata::VPoint &p) {
        return !(p.x >= cube_in_min_x_ && p.x <= cube_in_max_x_ &&
                 p.y >= cube_in_min_y_ && p.y <= cube_in_max_y_ &&
                 p.z >= cube_in_min_z_ && p.z <= cube_in_max_z_) && (

                       p.x >= cube_out_min_x_ && p.x <= cube_out_max_x_ &&
                       p.y >= cube_out_min_y_ && p.y <= cube_out_max_y_ &&
                       p.z >= cube_out_min_z_ && p.z <= cube_out_max_z_) &&
               (p.x != sensor_pos_x_ || p.y != sensor_pos_y_, p.z != sensor_pos_z_);

    }

    void processLidarScan() {

        //clear scan
        l_scan_fused_.points.clear();
        l_scan_fused_.width = 0;
        l_scan_fused_.header.seq = scan_no_;
        l_scan_fused_.header.frame_id = "base_footprint";
        l_scan_fused_.header.stamp = l_scan_strongest_.header.stamp;

        // transform clouds to world frame for insertion
        pcl::transformPointCloud(l_scan_strongest_, l_scan_strongest_, velo_to_footprint_);
        pcl::transformPointCloud(l_scan_last_, l_scan_last_, velo_to_footprint_);

        // Iterate through velodyne scan
        for (auto i = 0; i < l_scan_strongest_.width; ++i) {
            double dist;
            dist = sqrt(pow(l_scan_strongest_.points[i].x - l_scan_last_.points[i].x, 2) +
                        pow(l_scan_strongest_.points[i].y - l_scan_last_.points[i].y, 2) +
                        pow(l_scan_strongest_.points[i].z - l_scan_last_.points[i].z, 2));
            LidarPointFused p{};
            if (dist > 0) { // if echoes are different
                if (Formatter::checkScanLimits(l_scan_strongest_.points[i])) {
                    //Add strongest point
                    p.x = l_scan_strongest_.points[i].x;
                    p.y = l_scan_strongest_.points[i].y;
                    p.z = l_scan_strongest_.points[i].z;
                    p.ring = static_cast<uint8_t>(l_scan_strongest_.points[i].ring); // Ring
                    p.intensity = static_cast<uint8_t >(l_scan_strongest_.points[i].intensity); // Intensity
                    p.echo = 1;
                    l_scan_fused_.points.push_back(p);
                }
                if (Formatter::checkScanLimits(l_scan_last_.points[i])) {
                    // Add last point
                    p.x = l_scan_last_.points[i].x;
                    p.y = l_scan_last_.points[i].y;
                    p.z = l_scan_last_.points[i].z;
                    p.ring = static_cast<uint8_t>(l_scan_last_.points[i].ring); // Ring
                    p.intensity = static_cast<uint8_t >(l_scan_last_.points[i].intensity); // Intensity
                    p.echo = 2;
                    l_scan_fused_.points.push_back(p);
                }
            } else { // If echoes are the same
                if (Formatter::checkScanLimits(l_scan_strongest_.points[i])) {
                    p.x = l_scan_strongest_.points[i].x;
                    p.y = l_scan_strongest_.points[i].y;
                    p.z = l_scan_strongest_.points[i].z;
                    p.ring = static_cast<uint8_t>(l_scan_strongest_.points[i].ring); // Ring
                    p.intensity = static_cast<uint8_t >(l_scan_strongest_.points[i].intensity); // Intensity
                    p.echo = 0;
                    l_scan_fused_.points.push_back(p);
                }
            }
        }
        scan_no_++;
        pubPcl_.publish(l_scan_fused_);
    }

protected:
    ros::NodeHandle n_;
    ros::Publisher pubPcl_;

    Eigen::Matrix4f velo_to_footprint_;

    velodyne_rawdata::VPointCloud l_scan_strongest_; // last scan from velodyne
    velodyne_rawdata::VPointCloud l_scan_last_; // last scan from velodyne


    LidarScanFused l_scan_fused_;
    unsigned int scan_no_;

    float sensor_pos_x_;
    float sensor_pos_y_;
    float sensor_pos_z_;

    // PCL Filters
    float cube_out_min_x_;
    float cube_out_max_x_;
    float cube_out_min_y_;
    float cube_out_max_y_;
    float cube_out_min_z_;
    float cube_out_max_z_;

    // Cube Filters
    float cube_in_min_x_;
    float cube_in_max_x_;
    float cube_in_min_y_;
    float cube_in_max_y_;
    float cube_in_min_z_;
    float cube_in_max_z_;

};

#endif // FORMATTER_H
