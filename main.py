import utility._open3d as _o3d
import utility._filepath as _fp
import utility._linalg3d as _la3
import rtabmap
import utility._RTS as _rts
import numpy as np

def viewcloud(centervec,toppoint , ang=135):
    return


def main(inputpath , outputpath):
    Orgpcd = _o3d.load(_fp.export(inputpath) + _fp.os.sep + 'Img_cloud.ply')
    poses = np.loadtxt(_fp.export(inputpath) + _fp.os.sep + '__camera_poses.txt')
    rgbPath = _fp.export(inputpath) + _fp.os.sep + 'Img_rgb' + _fp.os.sep

    depthPath = _fp.export(inputpath) + _fp.os.sep + 'Img_depth' + _fp.os.sep
    calibpath = _fp.export(inputpath) + _fp.os.sep + 'Img_calib' + _fp.os.sep

    campos = poses[:,1:4]
    RtsQuad = poses[:,4:-1]
    frontvec = np.asarray([0,0,1])
    for i in range(len(RtsQuad)):
        rotmat = _o3d.o3d.geometry.get_rotation_matrix_from_quaternion(RtsQuad[i])
        frontvec = frontvec + campos[i]
        frontvec = rotmat.T.dot(frontvec.T).T
        _o3d.o3d.visualization.draw_geometries([Orgpcd])
    return

if __name__ == '__main__':
    inputpath = outputpath = '/home/udit/Udit/Data/11thFloor'
    main(inputpath , outputpath)