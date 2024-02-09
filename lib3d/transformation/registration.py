import copy

import cv2
import numpy as np
from utility import _linalg3d as l3d
from utility import _open3d as o3d
from utility import _filepath as f
from PIL import Image
from pandas import read_fwf , read_csv
import yaml
import os
from pyquaternion import Quaternion
from skimage.transform import resize

class register():
    def __init__(self, inputdir, outputdir=None):
        self.inputdir = os.path.dirname(inputdir)
        self.outputdir = f.Fstatus(self.inputdir) if outputdir is None else outputdir
        self.outputdir = outputdir
        self.rgbpath = self.inputdir + os.sep + '__rgb'
        self.depthpath = self.inputdir + os.sep + '__depth'
        self.calibpath = self.inputdir + os.sep + '__calib'
        self.poses = read_csv(self.inputdir +os.sep+'__poses.txt', sep=" ")
        self.formatpose = 11 # for RGBD-SLAM Ros Mode
        colnames = ['timestamp' , 'x','y','z' , 'qx' , 'qy' ,'qz' ,'qw' , 'id']
        if self.poses.columns.shape[0] == 12 :
            ## posibily transformmat format
            colnames = ['r11','r12','r13','tx','r21','r22','r23' , 'ty' , 'r31','r32' , 'r33' , 'tz']
            self.formatpose = 0 # raw format rx tx ,ry ty , rz ,tz
        self.n_frames = self.poses['id'].values.astype(np.int16)
        print("Total Frames:{0}".format(self.n_frames.size))

    def get_images(self,fno,ext='.jpg'):
        #ext = '.'+os.listdir(self.rgbpath + os.sep)[0].split('.')[1]
        depth = np.asarray(Image.open(self.depthpath + os.sep + fno + '.png'))
        rgb = np.asarray(Image.open(self.rgbpath + os.sep + fno + ext))
        with open(self.calibpath + os.sep + fno + '.yaml', 'r') as f:
            for _ in range(2):f.readline()
            calib = yaml.safe_load(f)
        if rgb.shape[0]!= depth.shape[0]:
            calib['camera_matrix']['data'] = self._resize_cam_camtrix(np.reshape(calib['camera_matrix']['data'], (3, 3)),
                                                 scale=[depth.shape[1] / rgb.shape[1], depth.shape[0] / rgb.shape[0],1]).reshape(-1).tolist()
            rgb = resize(rgb , depth.shape)
        return (rgb , depth , calib)

    def _resize_cam_camtrix(self,matrix, scale=(1, 1, 1)):
        return np.multiply(matrix, np.asarray([scale]).T)

    def _get_transformation_matrix(self , f_no , mode='Quaternions'):
        q_ =self.poses.loc[self.poses['id'] == f_no, ['qw', 'qx', 'qy', 'qz']].values[0]
        quats = Quaternion(w=q_[0],x=q_[1],y=q_[2],z=q_[3])
        transformation_matrix = quats.normalised.transformation_matrix
        transformation_matrix[:3,:3] = transformation_matrix[:3,:3].T
        transformation_matrix[:3,-1] = -self.poses.loc[self.poses['id'] == f_no, ['x', 'y', 'z']].values[0]
        return transformation_matrix

    def handle_node(self,fno):
        rgb, depth, calib = self.get_images(fno=str(fno))
        intrinsic = np.reshape(calib['camera_matrix']['data'], (3, 3))
        if rgb.shape[0] != depth.shape[0]:intrinsic = self._resize_cam_camtrix(intrinsic,scale=[depth.shape[1] / rgb.shape[1], depth.shape[0] / rgb.shape[0],1])
        points, colors = o3d.depth2pts(depth=depth, intrinsic=intrinsic, rgb=rgb , roundup=4)
        pcd = o3d._topcd(points=points, colors=colors)
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location((0, 0, 0))
        transform_mat = self._get_transformation_matrix(fno)

        return pcd , transform_mat

    def estimate_rgb_(self ,src_img , targ_img):
        initT = np.identity(4)
        success = False

        orb = cv2.ORB.create(scaleFactor=1.2,nlevels=8,edgeThreshold=31,
                             firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,nfeatures=100,patchSize=31)
        [kp_s, des_s] = orb.detectAndCompute(src_img, None)
        [kp_t, des_t] = orb.detectAndCompute(targ_img, None)
        if not (len(kp_s) and len(kp_t)):
            # no relation / correspondance
            return success, initT
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_s, des_t)
        pts_src = np.asarray([kp_s[m.queryIdx].pt for m in matches])
        pts_targ = np.asarray([kp_t[m.trainIdx].pt for m in matches])
        k=1

        return

    def estimate_xyz_(self):
        initT = np.identity(4)
        return

    def register(self, method='point2plane', voxel_size=0.01):
        startf = 0
        stopf = 15#len(self.n_frames)
        skipf = 1
        correction = np.zeros((4,4))
        max_corres_dist = .2 #should be in 0.1 to 0.01 or lower for better
        finalpcd = o3d._topcd()
        for i_ in range(startf,stopf,skipf):
            print(f"frame:{i_}/{(stopf - startf)}")
            prevfno , currfno = self.n_frames[i_ - 1] , self.n_frames[i_]
            curr_pcd, curr_Tfm = self.handle_node(currfno)
            curr_pcd.transform(curr_Tfm)
            if i_ == startf:
                finalpcd = copy.deepcopy(curr_pcd)
                continue
            prev_pcd , prev_Tfm  = self.handle_node(prevfno)
            #prev_Tfm = prev_Tfm if correction[0,0] == 0 else correction
            prev_pcd.transform(prev_Tfm)
            if i_ == stopf:
                finalpcd += curr_pcd
                break

            if 0:
                method1 = o3d.o3d.pipelines.registration.TransformationEstimationPointToPlane()
                criteria1 = o3d.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
                reg_p2l = o3d.o3d.pipelines.registration.registration_icp(curr_pcd.voxel_down_sample(voxel_size),
                                                                          finalpcd.voxel_down_sample(voxel_size),
                                                                          max_correspondence_distance=max_corres_dist,
                                                                          init=np.eye(4),
                                                                          estimation_method=method1,
                                                                          criteria=criteria1)
                curr_pcd.transform(reg_p2l.transformation)

                print("correction:",reg_p2l.transformation - curr_Tfm)
                correction = reg_p2l.transformation

            if 1:
                #o3d.NormalViz([copy.copy(curr_pcd).paint_uniform_color((0,0,0))] + [finalpcd])
                method = o3d.o3d.pipelines.registration.TransformationEstimationForColoredICP()
                criteria1 = o3d.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                reg_p2l = o3d.o3d.pipelines.registration.registration_colored_icp(
                    copy.copy(curr_pcd).voxel_down_sample(voxel_size),
                    copy.copy(finalpcd).voxel_down_sample(voxel_size),
                    max_correspondence_distance=max_corres_dist,
                    init=np.eye(4),
                    estimation_method=method,
                    criteria=criteria1)
                #if len(np.asarray(reg_p2l.correspondence_set)):
                curr_pcd.transform(reg_p2l.transformation)
                correction = reg_p2l.transformation

            finalpcd += curr_pcd
            #o3d.NormalViz([finalpcd])
        o3d.NormalViz([finalpcd])
        return