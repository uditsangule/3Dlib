# Author :udit
# Created on : 21/03/24
# Features :
import numpy as np
import open3d.cpu.pybind.geometry
from skspatial.objects import Line as LL

class Line():
    """defining line as equation ; infinite line"""

    def __init__(self, vector, point):
        if type(vector) == list: vector = np.asarray(vector)
        self.denorm = np.linalg.norm(vector)
        self.vector = vector / self.denorm
        self.direction = self.vector
        if type(point) == list: point = np.asarray(point)
        self.point = point
        self.d = -np.dot(self.vector, self.point)
        self.dimension = self.point.size
        self.l = None

    def get_point(self, t=1):
        return self.point + self.vector * t

    @classmethod
    def best_fit(cls,points ,MAX_POINTS=10**4):
        if type(points) != np.ndarray : points = np.asarray(points)
        if points.shape[0] > MAX_POINTS : points = points[np.random.randint(points.shape[0] , size=MAX_POINTS)]
        center = np.mean(points , axis=0)
        eig_v , eig_vector = np.linalg.eig(np.cov(points.T))
        direction = eig_vector[:,np.argmax(eig_v)]
        return cls(point=center , vector=-direction)

    def project_points(self , points):
        if points.ndim == 1 : points = points[:,np.newaxis]
        dot_ = np.dot(points - self.point , self.vector)
        return self.point + dot_[:,np.newaxis] * self.vector

    def distance_points(self,points):
        return np.linalg.norm(points - self.project_points(points) , axis=1)


    def intersect_line(self , other):
        """vec_perpendicular = np.cross(self.vector , other.vector)
        vec = np.cross(self.point - other.point , other.vector).dot(vec_perpendicular)
        return self.point + vec/np.linalg.norm(vec_perpendicular)**2 * self.vector"""
        return
    def __repr__(self):
        return self.__str__()

    def plot(self, t=1):
        p = open3d.utility.Vector3dVector(np.asarray([self.get_point(t=-t), self.get_point(t=t)]))
        l = open3d.utility.Vector2iVector([[0, 1]])
        return open3d.cpu.pybind.geometry.LineSet(points=p, lines=l)

    def __str__(self):
        return f"Line Normal:{self.vector} Centroid:{self.point} denorm:{self.denorm} ,d:{self.d}"


class LineSegment(Line):
    """
    defining line as equations , finite with two end points
    """

    def __init__(self, vector, point=None, endpoints=None):
        if endpoints is None and point is None:
            raise print("point and endpoints both cannot be None!")
        super().__init__( vector, point)
        self.l = np.linalg.norm(endpoints[1] - endpoints[0])
        return
