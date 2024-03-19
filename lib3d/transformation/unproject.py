import copy

import cv2
import numpy as np
from utility import _linear_algebra as l3d
from utility import _open3d as o3d
from utility import _filepath as f
from PIL import Image
from pandas import read_fwf , read_csv
import yaml
import os
from pyquaternion import Quaternion
from skimage.transform import resize


class unproject:
    def __init__(self , pointcloudpath , calibration_path ,viz=1):
        self.pointcloud  = o3d.load(pointcloudpath)
        self.viz = viz
        return

    def get_direction_vec(self):
        loc = np.random.rand(1,3)
        dir = np.random.rand(1,3)
        return

    def get_rays(self , fov=70):
        return

