import numpy as np


import pyquaternion as pyqr
import vg

xaxis , yaxis ,zaxis = np.eye(3)

def get_rotmat(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: source vector
    :param vec2: target vector on which the source vector will be rotated
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1/np.linalg.norm(vec1) if np.linalg.norm(vec1)!=1 else vec1
    b = vec2/np.linalg.norm(vec2) if np.linalg.norm(vec2)!=1  else vec2
    v = np.cross(a, b)
    if not any(v): return np.eye(3)
    #if not all zeros then
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

