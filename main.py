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
import pandas as pd
import tqdm
from scipy.spatial.transform import Rotation
import yaml

def normalizeQuats(Quats , tol = 1e-7 ):
    denorm = np.linalg.norm(Quats , axis=1)
    idx = np.where(np.abs( denorm - 1) > tol)[0]
    if len(idx) < 1 : return Quats
    Quats[idx] = Quats[idx]/np.repeat(denorm[idx] , repeats=4 , axis=0).reshape((len(idx),4))
    return Quats

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

    mask = np.asarray(
        Image.open( inputpath + os.sep + 'Segmentation/Img_masks/' + str(fid) + '.png'))

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

    return depth, rgb, calib , mask


def main(inputpath, outputpath):
    Orgpcd = _o3d.load(_fp.export(inputpath) + _fp.os.sep + "Img_cloud.ply")
    poses = np.loadtxt(_fp.export(inputpath) + _fp.os.sep + "__camera_poses.txt")
    classmeta = pd.read_csv(filepath_or_buffer='./data/coco_classmeta.csv')

    rgbPath = _fp.export(inputpath) + _fp.os.sep + "Img_rgb" + _fp.os.sep
    depthPath = _fp.export(inputpath) + _fp.os.sep + "Img_depth" + _fp.os.sep
    calibpath = _fp.export(inputpath) + _fp.os.sep + "Img_calib" + _fp.os.sep

    campos = poses[:, 1:4]
    RtsQuad = poses[:, 4:-1]
    RtsQuad = normalizeQuats(RtsQuad)
    pcd = _o3d._topcd()
    Allpts = np.zeros(shape=(1, 3))
    Allcol = np.zeros(shape=(1, 3))
    for i_, fid in enumerate(poses[:, -1].astype(np.int16)):
        print(
            f"frame:{i_}/{len(poses)} , progress: {round((i_ + 1)*100/len(poses) , 2)}%",
            end="\r",
        )
        rotmat = Rotation.from_quat(RtsQuad[i_])
        depth, rgb, calib , mask = _get_data(inputpath, fid)
        orgpts, colors = _o3d.getdepth2pcd(depth, rgb, calib , mask , classmeta)
        orgpts = rotmat.apply(orgpts)
        orgpts = orgpts + campos[i_]
        Allpts = np.append(Allpts, orgpts, axis=0)
        Allcol = np.append(Allcol, colors, axis=0)

        # pcd += _o3d._topcd(points=orgpts, colors=colors)
    Allpts, indx ,counts = np.unique(Allpts.round(3), return_index=1 , return_counts=1, axis=0)
    Allcol = Allcol[indx]
    print(Allpts.shape)
    Allpcd = _o3d._topcd(points=Allpts, colors=Allcol)
    _o3d.NormalViz([Allpcd], _ui=False)

    return


if __name__ == "__main__":
    inputpath = outputpath = "/home/udit/Udit/Data/Tg_Home"
    cpf.StartCodeProfiler()
    main(inputpath, outputpath)
    print("\n")
    cpf.DisplayCodeProfilerResultsAndStopCodeProfiler(top=10)
