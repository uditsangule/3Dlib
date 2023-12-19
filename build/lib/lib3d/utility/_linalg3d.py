import numpy as np
from skspatial.objects import Plane , Line , Vector

def Point2PointDist(points , ref , positive=True):
    res_ = np.linalg.norm((points - ref) , axis=1)
    # Euclidian Distance which is always positive!
    return res_

def toPointDistance(P , points , positive=True):
    """
    Returns Distance of Points from P ['Plane' , 'Line']
    Args:
        P: Plane / Line equation in skspatial format i.e Plane.vector , Plane.point
        points: [N,3] points in 3D or 2D
        positive: flag to return only positives or with +/-. default is True

    Returns: Distance of points from P

    """
    P.vector = P.vector.unit()
    res_ = (points - P.point).dot(P.vector)
    return res_ if not positive else np.abs(res_)

def project( points , plane=None , line=None):
    """
    projects N points onto plane or line whichever is given
    Args:
        points: [N,3] points in 3D,2D space
        plane: Equation of plane in ['centroid','normal'] format
        line: Equation of line in ['centroid','direction'] format

    Returns: Projected points onto the Shape [ plane / line ]

    """
    if plane is None and line is None : print('No source of projection given!'); return None
    Proj_on = plane if line is None else line
    Proj_on.vector = Proj_on.vector.unit()
    D = toPointDistance(P=Proj_on , points=points , positive=False)
    ## if line projection
    if plane is None : return Proj_on.point + D.reshape((len(D),1))*Proj_on.vector
    ## else plane projection
    return points - D.reshape((len(points), 1)) * Proj_on.vector



def findBestPlane(points , Maxpoints = 10^5):
    """
    Function to best fit 3D points to a Plane
    Args:
        points: (N,3) points in 3D space
        Maxpoints: Maximum random points to be considered while fitting

    Returns: Plane Equation of fitted plane. in Vector and Centroid of plane

    """
    if type(points) == 'list': points = np.asarray(points)
    if len(points) > Maxpoints: points = points[np.random.randint(len(points),size=Maxpoints)]
    # get eigenvalues and eigenvectors of covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(np.cov((points - np.mean(points, axis=0)).T))
    # select eigenvector with smallest eigenvalue
    normal = eig_vecs[:, np.argmin(eig_vals)]
    return Plane(point=np.mean(points, axis=0).round(4) , normal=-normal.round(4))

