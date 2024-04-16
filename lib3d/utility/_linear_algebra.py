import numpy as np
from ._plane import Plane , PlaneSegment
from ._line import Line , LineSegment

def tounit(vec: np.ndarray) -> np.ndarray:
    """converts into unitvectors , normalized form"""
    if vec.ndim == 1: return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


def vec_angle(svec, tvec, normalize=True, maxang=180, signed=False, units="deg") -> np.ndarray:
    """
    Calculates the angles in rad or deg between source vectors to target vector
     Args:
        svec: (3) source vector
        tvec: (N,3) or (3) target vector/s from where to check angle
        normalize: True if normalization is needed to be done on vectors
        maxang: outputs in [0,maxang] range. Usually [0,90] default it is [0,180]
        signed: if clock or anticlockwise angles needed
        units: format of unit required as output, "rad": in radians, "deg": in degrees (default).

    Returns: (N) array of angles between vectors.
    """
    if isinstance(svec, (list, tuple)):
        svec = np.asarray(svec)
    if isinstance(tvec, (list, tuple)):
        tvec = np.asarray(tvec)

    if normalize:
        svec = tounit(svec)
        tvec = tounit(tvec)
    dotprod = np.dot(tvec, svec)
    ang = np.degrees(np.arccos(np.clip(dotprod, -1.0, 1.0)))
    if signed is not None:
        dp = np.dot(tvec, signed)
        k = 1  # code for assigning sign in terms of look at vector. will code later!
    if maxang == 90: ang = np.where(ang > maxang, 180 - ang, ang)
    if units in 'rad': return np.radians(ang).astype(np.float16)
    return ang.astype(np.float16)


def Point2PointDist(points: np.ndarray, ref: np.ndarray, positive=True) -> np.ndarray:
    # Euclidian Distance which is always positive!
    return np.linalg.norm((points - ref), axis=1)


def Plane2PlaneDist(plane1, Otherplanes, positive=True) -> np.ndarray:
    dot_ = np.asarray(plane1[0].vector)[:, np.newaxis].T.dot(np.vstack([p.vector for p in Otherplanes]).T)[0].round(3)
    n_ = np.cross(plane1.vector, np.vstack([n.vector for n in Otherplanes]))
    n_ = n_ / np.linalg.norm(n_, axis=1, keepdims=True)
    res_ = (np.vstack([p.point for p in Otherplanes]) - plane1.point).dot(n_)
    return np.abs(res_) if positive else res_


def toPointDistance(P, points: np.ndarray, positive=True) -> np.ndarray:
    """
    Returns Distance of Points from P ['Plane' , 'Line']
    Args:
        P: Plane / Line equation in skspatial format i.e. Plane.vector , Plane.point
        points: [N,3] points in 3D or 2D
        positive: flag to return only positives or with +/-. default is True

    Returns: Distance of points from P

    """
    P.unit()
    res_ = (points - P.point).dot(P.vector)
    return res_ if not positive else np.abs(res_)


def project(points: np.ndarray, plane=None, line=None) -> np.ndarray or None:
    """
    projects N points onto plane or line whichever is given
    Args:
        points: [N,3] points in 3D,2D space
        plane: Equation of plane in ['centroid','normal'] format
        line: Equation of line in ['centroid','direction'] format

    Returns: Projected points onto the Shape [ plane / line ]

    """
    if plane is None and line is None: print('No source of projection given!'); return None
    Proj_on = plane if line is None else line
    Proj_on.vector = Proj_on.vector.unit()
    D = toPointDistance(P=Proj_on, points=points, positive=False)
    ## if line projection
    if plane is None: return Proj_on.point + D.reshape((len(D), 1)) * Proj_on.vector
    ## else plane projection
    return points - D.reshape((len(points), 1)) * Proj_on.vector


def findBestPlane(points: np.ndarray, Maxpoints: int = 10 ^ 5):
    """
    Function to best fit 3D points to a Plane
    Args:
        points: (N,3) points in 3D space
        Maxpoints: Maximum random points to be considered while fitting

    Returns: Plane Equation of fitted plane. in Vector and Centroid of plane

    """
    if type(points) == 'list': points = np.asarray(points)
    if len(points) > Maxpoints: points = points[np.random.randint(len(points), size=Maxpoints)]
    # get eigenvalues and eigenvectors of covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(np.cov((points - np.mean(points, axis=0)).T))
    # select eigenvector with smallest eigenvalue
    normal = eig_vecs[:, np.argmin(eig_vals)]
    return Plane(point=np.mean(points, axis=0).round(4), vector=-normal.round(4))
