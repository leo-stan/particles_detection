//File name: bag_formatter.cpp
//Author: Leo Stanislas
//Date created: 2018/06/20
//Date last modified: 2018/06/29

#include <smoke_detection/bag_formatter.h>

#include <std_msgs/Int32.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

BagFormatter::BagFormatter() :
        root_dir_("/home/leo/phd/smoke_detection/src/smoke_detection/data/rosbags/") {


    loadBagList();
    std::string path_out = root_dir_ + bag_out_name_;
    bag_out_.open(path_out, rosbag::bagmode::Write);

    velo_to_footprint_ << 0, 1, 0, sensor_pos_x_,
            -1, 0, 0, sensor_pos_y_,
            0, 0, 1, sensor_pos_z_,
            0, 0, 0, 1;

    pubPcl_ = n_.advertise<LidarScanFused>("/lidar_points_filtered", 10);

}

void BagFormatter::formatBags() {

    rosbag::Bag bag;

    std::vector<std::string> topics;
    topics.emplace_back("/velodyne_points_dual");
    topics.emplace_back("/velodyne_points");
    topics.emplace_back("/tf");
//    topics.push_back(std::string("/multisense"));

    std_msgs::Int32 label;

    bool l_strong_flag = false;
    bool l_last_flag = false;

    std::cout << "Writting data in " << bag_out_name_ << std::endl;

    for (auto input_bag = bag_list_.begin(); input_bag < bag_list_.end(); ++input_bag) {

        std::cout << "Processing bag: " << input_bag->bag_name << " label: " << input_bag->label << std::endl;

        std::string path = root_dir_ + input_bag->bag_name;
        bag.open(path);

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        BOOST_FOREACH(rosbag::MessageInstance const m, view) {
                        if (m.getTime().sec >= input_bag->t_start && m.getTime().sec < input_bag->t_end) {


                            sensor_msgs::PointCloud2::ConstPtr s = m.instantiate<sensor_msgs::PointCloud2>();
                            if (s != NULL) {
                                if (m.getTopic() == "/velodyne_points_dual") {
                                    pcl::fromROSMsg(*s, l_scan_strongest_);
                                    l_strong_flag = true;
                                } else if (m.getTopic() == "/velodyne_points") {
                                    pcl::fromROSMsg(*s, l_scan_last_);
                                    l_last_flag = true;
                                }
//                                else{
//                                    bag_out_.write(m.getTopic(),m.getTime(),m);
//                                }

                                if (l_strong_flag && l_last_flag) {
                                    if (l_scan_strongest_.header.seq == l_scan_last_.header.seq) {
                                        ROS_DEBUG("seq: %d", l_scan_strongest_.header.seq);
                                        updateCubes(*input_bag);
                                        processLidarScan();
                                        label.data = input_bag->label;
                                        bag_out_.write("lidar_points_filtered", ros::Time::now(), l_scan_fused_);
                                        bag_out_.write("label", ros::Time::now(), label);
                                    } else
                                        ROS_ERROR("Bag Formatter: Lidar scans (strong/last) have different seq IDs");
                                    l_strong_flag = false;
                                    l_last_flag = false;
                                }

                            }
//                            else{
//                                bag_out_.write(m.getTopic(),m.getTime(),m);
//                            }

                        }
                    }

        bag.close();
    }

    bag_out_.close();
}

void BagFormatter::updateCubes(const BagFormatter::BagInput &bag) {

    cube_in_min_x_ = bag.cube_in_min_x;
    cube_in_max_x_ = bag.cube_in_max_x;
    cube_in_min_y_ = bag.cube_in_min_y;
    cube_in_max_y_ = bag.cube_in_max_y;
    cube_in_min_z_ = bag.cube_in_min_z;
    cube_in_max_z_ = bag.cube_in_max_z;
    cube_out_min_x_ = bag.cube_out_min_x;
    cube_out_max_x_ = bag.cube_out_max_x;
    cube_out_min_y_ = bag.cube_out_min_y;
    cube_out_max_y_ = bag.cube_out_max_y;
    cube_out_min_z_ = bag.cube_out_min_z;
    cube_out_max_z_ = bag.cube_out_max_z;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bag_formatter");
    ros::Time::init();

    BagFormatter bag_formatter;
    bag_formatter.formatBags();

    return 0;
}
