# particles_detection


## Dependencies

- numpy 
- torch
- tensorboardX
- rospy
- easydict
- glob

## Installation

Clone this repository in a ROS workspace

## Files

### Traditional ML approach (Random Forest)

- LidarDatasetHC.py
- train_model.py
- topic_prediction.py

### Deep Learning approach

- LidarDataset.py
- train_smokenet.py
- smokenet_topic_prediction.py

## Data Preparation

- bag_formatter: Select parts of data in 3D space and put a label on it
- extract_rosbags: Take in formatted bags and split in training/testing/validation sets

### Steps
1. Select bags of data in bag_formatter.h
2. Run bag_formatter.cpp
3. Select formatted bags in extract_rosbags.py
4. Run extract_rosbags.py

## Train

train_model.py or train_smokenet.py

with input_model = None

## Evaluate

train_model.py or train_smokenet.py

with input_model = model_file_name

## ROS Topic prediction

'''bash
$rosparam set use_sim_time true
$roslaunch smoke_detection transforms.launch
$rosrun smoke_detection scan_formatter
$rosbag play whatever bag you want to predict
$rosrun smoke_detection topic_prediction or smokenet_topic_prediction
'''

## TODO

- Add link to dataset
