# Author :udit
# Created on : 21/03/24
# Features :
import numpy as np


class Line():
    """defining line as equation ; infinite line"""

    def __init__(self, vector, point):
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Vector cannot be zero.")
        self.vector = vector / norm  # Normalize
        self.norm = norm
        self.direction = self.vector
        self.point = point
        self.d = -np.dot(self.vector, self.point)
        self.dimension = self.point.size
        self.l = None

    def get_point(self, t=1):
        t = np.asarray(t, dtype=np.float64)
        return self.point + np.outer(t, self.vector)

    @classmethod
    def best_fit(cls, points, MAX_POINTS=10 ** 4):
        if type(points) != np.ndarray: points = np.asarray(points)
        if points.shape[0] > MAX_POINTS: points = points[np.random.randint(points.shape[0], size=MAX_POINTS)]
        center = np.mean(points, axis=0)
        # eig_v , eig_vector = np.linalg.eig(np.cov(points.T))
        # direction = eig_vector[:,np.argmax(eig_v)]
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        direction = vh[0]  # First singular vector (dominant direction)
        return cls(point=center, vector=-direction)

    def project_points(self, points):
        dot_ = np.dot(points - self.point, self.vector)
        return self.point + np.outer(dot_, self.vector)

    def distance_points(self, points):
        proj_points = self.project_points(points)
        return np.linalg.norm(points - proj_points, axis=1)

    def intersect_line(self, other):
        """vec_perpendicular = np.cross(self.vector , other.vector)
        vec = np.cross(self.point - other.point , other.vector).dot(vec_perpendicular)
        return self.point + vec/np.linalg.norm(vec_perpendicular)**2 * self.vector"""
        return

    def __repr__(self):
        return self.__str__()

    def plot(self, t=1):
        p = np.asarray([self.get_point(t=-t), self.get_point(t=t)])
        l = np.asarray([[0, 1]])
        return p, l

    def __str__(self):
        return f"Line Normal:{self.vector} Centroid:{self.point} denorm:{self.denorm} ,d:{self.d}"


class LineSegment(Line):
    """Defines a finite line segment between two points or using center, direction, and length."""

    def __init__(self, p1=None, p2=None, center=None, vector=None, length=None):
        if p1 is not None and p2 is not None:
            p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
            vector = p2 - p1
            center = (p1 + p2) / 2
            length = np.linalg.norm(vector)
        elif center is not None and vector is not None and length is not None:
            center = np.asarray(center, dtype=np.float64)
            vector = np.asarray(vector, dtype=np.float64) / np.linalg.norm(vector)
        else:
            raise ValueError("Provide either (p1, p2) OR (center, vector, length).")

        self.length = length
        super().__init__(vector, center)

    def clamp_to_segment(self, points):
        """
        Clamp projected points to lie within segment limits.
        """
        points = np.asarray(points, dtype=np.float64)
        t = np.dot(points - self.point, self.vector)
        t = np.clip(t, -self.length / 2, self.length / 2)  # Clamp between [-L/2, L/2]
        return self.point + np.outer(t, self.vector)

    def distance_points(self, points):
        """
        Compute shortest distance from given points to the line segment.
        """
        points = np.asarray(points, dtype=np.float64)
        clamped_points = self.clamp_to_segment(self.project_points(points))
        return np.linalg.norm(points - clamped_points, axis=1)

    def __str__(self):
        return f"LineSegment: Center={self.point}, Direction={self.vector}, Length={self.length}"
