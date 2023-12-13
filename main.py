import utility._open3d as _o3d
import utility._opencv as _cv2
import utility._filepath as _fp
import utility._linalg3d as _la3
import rtabmap
import utility._RTS as _rts
from PIL import Image
import utility.CodeProfiler as cpf
import numpy as np
import os
import tqdm
import yaml


def _get_data(inputpath, fid):
    rgb = np.asarray(
        Image.open(
            _fp.export(inputpath)
            + _fp.os.sep
            + "Img_rgb"
            + _fp.os.sep
            + str(fid)
            + ".jpg"
        )
    )
    depth = np.asarray(
        Image.open(
            _fp.export(inputpath)
            + _fp.os.sep
            + "Img_depth"
            + _fp.os.sep
            + str(fid)
            + ".png"
        )
    )
    with open(
        _fp.export(inputpath)
        + _fp.os.sep
        + "Img_calib"
        + _fp.os.sep
        + str(fid)
        + ".yaml"
    ) as infile:
        for i in range(2):
            _ = infile.readline()
        calib = yaml.safe_load(infile)
    return depth, rgb, calib


def main(inputpath, outputpath):
    Orgpcd = _o3d.load(_fp.export(inputpath) + _fp.os.sep + "Img_cloud.ply")
    poses = np.loadtxt(_fp.export(inputpath) + _fp.os.sep + "__camera_poses.txt")

    rgbPath = _fp.export(inputpath) + _fp.os.sep + "Img_rgb" + _fp.os.sep
    depthPath = _fp.export(inputpath) + _fp.os.sep + "Img_depth" + _fp.os.sep
    calibpath = _fp.export(inputpath) + _fp.os.sep + "Img_calib" + _fp.os.sep

    campos = poses[:, 1:4]
    RtsQuad = poses[:, 4:-1]
    pcd = _o3d._topcd()
    for i_, fid in enumerate(poses[:, -1].astype(np.int16)):
        print(f'frame:{i_}/{len(poses)} , progress: {round((i_ + 1)*100/len(poses) , 2)}%' , end='\r')
        rotmat = _o3d.o3d.geometry.get_rotation_matrix_from_quaternion(RtsQuad[i_])
        depth, rgb, calib = _get_data(inputpath, fid)
        orgpts, colors = _o3d.getdepth2pcd(depth, rgb, calib)
        orgpts = np.asarray(_o3d._topcd(points=orgpts).rotate(rotmat).points)
        orgpts = orgpts + campos[i_]
        pcd += _o3d._topcd(points=orgpts , colors=colors)
        #_o3d.NormalViz([pcd] , _ui=0)

    return


if __name__ == "__main__":
    inputpath = outputpath = "/home/udit/Udit/Data/11thFloor"
    cpf.StartCodeProfiler()
    main(inputpath, outputpath)
    print('\n');cpf.DisplayCodeProfilerResultsAndStopCodeProfiler(top=10)
