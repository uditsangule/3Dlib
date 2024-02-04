import copy

import numpy as np
from utility import _linalg3d as l3d
from utility import _open3d as o3d
from utility import _opencv as cv
from utility import _filepath as f
from PIL import Image
from pandas import read_fwf , read_csv
import yaml
import os
from pyquaternion import Quaternion
class register():
    def __init__(self, inputdir, outputdir=None):
        self.inputdir = os.path.dirname(inputdir)
        self.outputdir = f.Fstatus(self.inputdir) if outputdir is None else outputdir
        self.outputdir = outputdir
        self.rgbpath = self.inputdir + os.sep + '__rgb'
        self.depthpath = self.inputdir + os.sep + '__depth'
        self.calibpath = self.inputdir + os.sep + '__calib'
        self.poses = read_csv(self.inputdir +os.sep+'__poses.txt', sep=" ")
        self.n_frames = self.poses['id'].values.astype(np.int16)
        print("Total Frames:{0}".format(self.n_frames.size))

    def get_images(self,fno,ext='.jpg'):
        #ext = '.'+os.listdir(self.rgbpath + os.sep)[0].split('.')[1]
        rgb = np.asarray(Image.open(self.rgbpath + os.sep + fno + ext))
        depth = np.asarray(Image.open(self.depthpath + os.sep + fno +'.png'))
        with open(self.calibpath + os.sep + fno + '.yaml', 'r') as f:
            for _ in range(2):f.readline()
            calib = yaml.safe_load(f)
        return (rgb , depth , calib)

    def _resize_cam_camtrix(self,matrix, scale=(1, 1, 1)):
        return np.multiply(matrix, np.asarray([scale]).T)

    def _get_transformation_matrix(self , f_no , mode='Quaternions'):
        quats = Quaternion(self.poses.loc[self.poses['id'] == f_no, ['qw', 'qy', 'qz', 'qz']].values[0])
        transformation_matrix = quats.normalised.transformation_matrix
        transformation_matrix[:3,-1] = self.poses.loc[self.poses['id'] == f_no, ['x', 'y', 'z']].values[0]
        return transformation_matrix

    def handle_node(self,fno):
        rgb, depth, calib = self.get_images(fno=str(fno))
        intrinsic = np.reshape(calib['camera_matrix']['data'], (3, 3))
        intrinsic = self._resize_cam_camtrix(intrinsic,
                                             scale=[depth.shape[1] / rgb.shape[1], depth.shape[0] / rgb.shape[0],
                                                    1])
        points, colors = o3d.depth2pts(depth=depth, intrinsic=intrinsic, rgb=rgb)
        pcd = o3d._topcd(points=points, colors=colors)
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location((0, 0, 0))
        transform_mat = self._get_transformation_matrix(fno)
        return pcd , transform_mat

    def register(self, method='point2plane', voxel_size=0.01):
        pcd , transform_matrix = self.handle_node(fno=self.n_frames[0])
        correction = np.zeros(transform_matrix.shape)

        pcd.transform(transform_matrix)
        prev_pcd = copy.deepcopy(pcd)
        prev_temp = copy.deepcopy(pcd.voxel_down_sample(voxel_size))
        max_corres_dist = 0.05
        finalpcd = copy.deepcopy(pcd)
        for i_ , fno in enumerate(self.n_frames[1:]):
            print(f"frame:{fno}/{len(self.n_frames)}")
            pcd , transform_matrix = self.handle_node(fno=fno)
            src_temp = copy.deepcopy(pcd.voxel_down_sample(voxel_size))
            transform_matrix = transform_matrix + correction

            reg_p2l = o3d.o3d.pipelines.registration.registration_icp(src_temp, prev_temp,
                                                                      max_correspondence_distance=max_corres_dist,
                                                                      init=transform_matrix,
                                                                      estimation_method=o3d.o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                                      criteria=o3d.o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                          max_iteration=1000))
            print(reg_p2l.fitness.__round__(3) , reg_p2l.inlier_rmse.__round__(3))
            if reg_p2l.fitness.__round__(3) < 0.15:continue
            if (reg_p2l.inlier_rmse > 0) and (reg_p2l.inlier_rmse < 0.1):
                reg_p2l = o3d.o3d.pipelines.registration.registration_colored_icp(src_temp, prev_temp,
                                                                                  max_correspondence_distance=max_corres_dist,
                                                                                  init=reg_p2l.transformation,
                                                                                  estimation_method=o3d.o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                                                                                  criteria=o3d.o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                                      max_iteration=1000))
                #correction += (reg_p2l.transformation - transform_matrix)/2
                transform_matrix = reg_p2l.transformation

            else:
                #correction += (reg_p2l.transformation - transform_matrix) / 2
                #transform_matrix = reg_p2l.transformation
                src_temp.transform(transform_matrix)
                prev_temp = copy.deepcopy(src_temp)
                continue
            pcd.transform(transform_matrix)
            src_temp.transform(transform_matrix)
            finalpcd += pcd
            prev_temp = copy.deepcopy(src_temp)
            o3d.NormalViz([finalpcd])
        return