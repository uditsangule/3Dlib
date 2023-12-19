import numpy as np
from skspatial.objects import Plane , Line , Vector

def DistancesPoints(points , refpoint , positive=True):
    res_ = np.linalg.norm((points - refpoint) , axis=1)
    return res_ if not positive else np.abs(res_)

def PlanetoPointDistance(plane , points , positive=True):
    res_ = (points - plane.point).dot(plane.vector)
    return res_ if not positive else np.abs(res_)

def LinetoPointDistance(line , points , pos=True):
    res_ = (points - line.point).dot(line.vector)
    return res_ if not pos else np.abs(res_)

def findBestPlane(points , Maxpoints = 10^5):
    """
    Function to best fit 3D points to a Plane
    Args:
        points: (N,3) points in 3D space
        Maxpoints: Maximum random points to be considered while fitting

    Returns: Plane Equation of fitted plane. in Vector and Centroid point of plane

    """
    if type(points) == 'list': points = np.asarray(points)
    if len(points) > Maxpoints: points = points[np.random.randint(len(points),size=Maxpoints)]
    # get eigenvalues and eigenvectors of covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(np.cov((points - np.mean(points, axis=0)).T))
    # select eigenvector with smallest eigenvalue
    normal = eig_vecs[:, np.argmin(eig_vals)]
    return Plane(point=np.mean(points, axis=0).round(4) , normal=-normal.round(4))

