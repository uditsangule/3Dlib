# Author :udit
# Created on : 28/03/24
# Features :
from __future__ import absolute_import
import os
import numpy as np
import yaml
from pandas import read_csv

from utility import _open3d as u3d
from utility import _opencv as ucv2
from utility import _filepath as fp
from utility import _linear_algebra as lgb
from utility import _RTS
from objectdetection._2d.yolov8 import yolov8


class register():
    def __init__(self, inputdir, outputdir=None, run_obj_detection=False, pose_format='raw', decimation=1):
        self.inputdir = os.path.dirname(inputdir)
        self.outputdir = fp.Fstatus(self.inputdir) if outputdir is None else fp.Fstatus(outputdir)
        self.rgbpath = self.inputdir + os.sep + '__rgb'
        self.depthpath = self.inputdir + os.sep + '__depth'
        self.calibpath = self.inputdir + os.sep + '__calib'
        self.decimation = decimation
        if pose_format == 'raw':
            # raw format 'raw' = 0
            cols = ['r11', 'r12', 'r13', 'tx', 'r21', 'r22', 'r23', 'ty', 'r31', 'r32', 'r33', 'tz']
            self.poses = read_csv(self.inputdir + os.sep + '__poses.txt', sep=" ", usecols=cols)
        if pose_format == 'slam':
            # rgdb-slam format = 11
            cols = ['#timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'id']
            self.poses = read_csv(self.inputdir + os.sep + '__poses.txt', sep=" ", usecols=cols)
        self.n_frames = self.poses['id'].values.astype(np.int16)
        self.yolo = None
        self.viz = u3d.o3d.visualization.Visualizer()

        if run_obj_detection:
            self.yolo = yolov8()
        return

    def fetch_images(self, frame_no):
        depth = ucv2.cv2.imread(self.depthpath + os.sep + f'{str(frame_no)}.png', ucv2.cv2.IMREAD_ANYDEPTH)
        if self.decimation > 1: depth = ucv2.pooling(depth, mode='mean', kernel=(self.decimation, self.decimation))
        rgb = ucv2.cv2.imread(self.rgbpath + os.sep + f'{str(frame_no)}.jpg', )
        objectmask = self.yolo.detect(frame=rgb) if self.yolo is not None else None
        normal_map = u3d.depth2normalmap(depth)
        with open(self.calibpath + os.sep + str(frame_no) + '.yaml', 'r') as f:
            for _ in range(2): f.readline()
            calib = yaml.safe_load(f)
        return (rgb, depth, objectmask, calib, normal_map)

    def fetch_data(self, frame_no, preprocess=True, object_detection=True):
        rgb, depth, objectmask, calib, normalmap = self.fetch_images(frame_no=frame_no)
        intrinsic = np.reshape(calib['camera_matrix']['data'], (3, 3))
        local_transform = np.reshape(calib['local_transform']['data'], (3, 4)).round(4)
        local_transform = np.append(local_transform, [[0, 0, 0, 1]], axis=0)
        # depth = ucv2.cv2.resize(depth , dsize=(rgb.shape[1],rgb.shape[0]) , interpolation=ucv2.cv2.INTER_NEAREST_EXACT)
        if rgb.shape[0] != depth.shape[0]:
            rgb2dpth_ratio = rgb.shape[0] / depth.shape[0]
            rgb = ucv2.cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=ucv2.cv2.INTER_NEAREST)
            intrinsic = intrinsic / rgb2dpth_ratio

        if preprocess:
            pcd, rgbd = u3d.images_topcd(depth_im=depth, rgb_im=rgb, intr=intrinsic, extr=local_transform)
            mesh = u3d.pcd2mesh(pcd, mode='pivot')
            # u3d.NormalViz([mesh])
            # facets = u3d.get_facets(mesh)
            return (rgb, depth, calib, normalmap, objectmask, rgbd, pcd, mesh)
        return (rgb, depth, calib, normalmap, objectmask)

    def fetch_pose(self, frame_no):
        if 'qw' in self.poses.keys():
            q = self.poses.loc[self.poses['id'] == frame_no, ['qw', 'qx', 'qy', 'qz']].values[0]
            t = self.poses.loc[self.poses['id'] == frame_no, ['x', 'y', 'z']].values[0]
            Tmatrix = _RTS.quats_to_Tmatrix(q)
            Tmatrix[:3, 3] = t
        elif 'tz' in self.poses.keys():
            # this part is depreciated as of now
            t1 = self.poses.loc[
                self.poses['id'] == frame_no, ['r11', 'r12', 'r13', 'tx', 'r21', 'r22', 'r23', 'ty', 'r31', 'r32',
                                               'r33', 'tz']].values[0]
            Tmatrix = np.append(np.reshape(t1, (3, 4)), [0, 0, 0, 1], axis=0)
        else:
            Tmatrix = np.eye(4)

        return Tmatrix

    def run(self, show=True, object_detection=True):
        finalpcd = u3d._topcd()
        finalmesh = u3d._tomesh()
        #self.viz.create_window(window_name='Registration' , width=1080 , height=720)
        #self.viz.add_geometry({'name':'finalpcd' , 'geometry':finalpcd})
        #self.viz.add_geometry(finalmesh)
        planerpatches = []
        for fno in self.n_frames:
            if fno > 10:
                break
            print(f"frameid:{fno}\r")
            _, _, _, _, _, rgbd, pcd, mesh = self.fetch_data(frame_no=fno, object_detection=object_detection)

            Tmatrix = self.fetch_pose(frame_no=fno)
            pcd.transform(Tmatrix)
            mesh.transform(Tmatrix)


            _, planemesh, planeqns, _ = u3d.detectplanerpathes(pcd, minptsplane=250, scale=(1.2, 1.2, 0.01))
            planerpatches.extend(planemesh)
            finalmesh += mesh
            finalpcd += pcd


            if show:
                u3d.NormalViz([finalmesh] + [u3d.axis_mesh(size=1)])
        self.viz.destroy_window()
        #u3d.NormalViz([finalmesh] + [u3d.axis_mesh(size=1)] + planerpatches)
        return
