#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""VoxelNet config system.
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C

# for dataset dir
__C.DATA_DIR = '/media/storage/leo-ws/voxelnet/sorted_data'
__C.CALIB_DIR = '/media/storage/leo-ws/voxelnet/data/training/calib'
__C.ROOT_DIR = '/home/leo/phd/smoke_detection/src/smoke_detection'

__C.Z_MIN = -10
__C.Z_MAX = 10
__C.Y_MIN = -10
__C.Y_MAX = 10
__C.X_MIN = -10
__C.X_MAX = 10
__C.VOXEL_X_SIZE = 0.2
__C.VOXEL_Y_SIZE = 0.2
__C.VOXEL_Z_SIZE = 0.2
__C.VOXEL_POINT_COUNT = 35
__C.MAP_TO_FOOTPRINT_X = 10
__C.MAP_TO_FOOTPRINT_Y = 10
__C.MAP_TO_FOOTPRINT_Z = 0
__C.VFE1_OUT = 32
__C.VFE2_OUT = 128
__C.VFE3_OUT = 256
__C.TIMING_COUNT_MAX = 50
