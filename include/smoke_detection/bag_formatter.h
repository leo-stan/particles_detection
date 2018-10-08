//File name: bag_formatter.h
//Author: Leo Stanislas
//Date created: 2018/06/20
//Date last modified: 2018/06/29

#ifndef BAG_FORMATTER_H
#define BAG_FORMATTER_H

#include <smoke_detection/formatter.h>

#include <rosbag/bag.h>

class BagFormatter : public Formatter {
public:

    struct BagInput {
        std::string bag_name;
        int label; // 0 hard, 1 smoke, 2 dust
        int t_start; // in sec
        int t_end; // in sec

        // Cube Filters
        float cube_in_min_x;
        float cube_in_max_x;
        float cube_in_min_y;
        float cube_in_max_y;
        float cube_in_min_z;
        float cube_in_max_z;

        float cube_out_min_x;
        float cube_out_max_x;
        float cube_out_min_y;
        float cube_out_max_y;
        float cube_out_min_z;
        float cube_out_max_z;

    };

    BagFormatter();

    void loadBagList() {

        bag_out_name_ = "bush.bag";

        // Hard stuff Bags

//        bag_list_.push_back(BagInput{"2018-04-10-12-14-25.bag", 0, 1523326519, 1523326523,
//                                     -3, 3, -3, 3, -5, 5,
//                                     -10, 10, -10, 10, -1, 3
//        });
//
//        bag_list_.push_back(BagInput{"2018-08-31-14-34-44.bag", 0, 1535690085, 1535690105,
//                                     0, 0, 0, 0, 0, 0,
//                                     -3, 10, -10, 3, -1, 3
//        });
        // Bush
        bag_list_.push_back(BagInput{"2018-08-31-14-18-44.bag", 0, 1535689125, 1535689132,
                                     0, 0, 0, 0, 0, 0,
                                     -30, 30, -30, 30, -1, 5
        });

        // Smoke Bags

//        bag_list_.push_back(BagInput{"2018-04-10-12-14-25.bag", 1, 1523326466, 1523326532,
//                                     0, 0, 0, 0, 0, 0,
//                                     0, 3, -2, 2, 0.5, 3
//        });
//
//        bag_list_.push_back(BagInput{"2018-08-31-13-46-29.bag", 1, 1535687234, 1535687262,
//                                     0, 0, 0, 0, 0, 0,
//                                     -3, 3, -3, 2, 0.5, 3
//        });
//
//            // Long range
//        bag_list_.push_back(BagInput{"2018-08-31-13-57-33.bag", 1, 1535687879, 1535687910,
//                                     0, 0, 0, 0, 0, 0,
//                                     -2, 8, -1.5, 1.5, 0.5, 3
//        });

        // Dust Bags

//        bag_list_.push_back(BagInput{"2018-08-31-14-15-25.bag", 2, 1535688941, 1535688988,
//                                     0, 0, 0, 0, 0, 0,
//                                     1, 7, -1, 4, 0.5, 3
//        });
//
//        bag_list_.push_back(BagInput{"2018-08-31-14-33-26.bag", 2, 1535690014, 1535690053,
//                                     0, 0, 0, 0, 0, 0,
//                                     -3, 7, -1, 4, 0.5, 3
//        });




    }

    void formatBags();

    void updateCubes(const BagFormatter::BagInput &bag);

private:

    rosbag::Bag bag_out_;
    std::string bag_out_name_;
    std::string root_dir_;
    std::vector<BagFormatter::BagInput> bag_list_;

};

#endif // BAG_FORMATTER_H
