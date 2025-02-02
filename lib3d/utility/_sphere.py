import numpy as np


class Sphere:
    def __init__(self, center, radius):
        """
        Initialize a Sphere object.

        Args:
        center (array-like): Coordinates of the sphere's center (2D or 3D).
        radius (float): Radius of the sphere.
        """
        self.center = np.array(center)
        self.radius = radius

    def distances(self, points):
        """
        Compute the Euclidean distance of points from the sphere's surface.

        Args:
        points (ndarray): An (N, D) array of N points in D dimensions (2D or 3D).

        Returns:
        distances (ndarray): Distance of each point from the sphere's surface.
        """
        points = np.asarray(points)  # Ensure input is an array
        distances = np.linalg.norm(points - self.center, axis=1) - self.radius
        return distances

    def classify_points(self, points):
        """
        Classify points as inside, on, or outside the sphere.

        Args:
        points (ndarray): An (N, D) array of N points in D dimensions (2D or 3D).

        Returns:
        labels (ndarray): Array of classification labels (-1: inside, 0: on, 1: outside).
        """
        distances = self.distances(points)
        labels = np.sign(distances)  # -1: inside, 0: on, 1: outside
        return labels

    def project_points(self, points):
        vectors_ = points - self.center
        norm_vecs = np.linalg.norm(vectors_, axis=1, keepdims=True)
        return self.center + (self.radius * vectors_ / norm_vecs)

    @classmethod
    def fit_sphere(cls, points):
        """
        Fit a sphere to a given set of points using least squares.

        Args:
        points (ndarray): An (N, D) array of N points in 2D or 3D.

        Returns:
        Sphere instance with optimized center and radius.
        """
        points = np.asarray(points)
        N, D = points.shape  # Number of points, dimension (2D or 3D)

        # Construct the linear system Ax = b
        A = np.hstack((2 * points, np.ones((N, 1))))  # [2x, 2y, 2z, 1]
        b = np.sum(points ** 2, axis=1)  # x^2 + y^2 + z^2

        # Solve the least squares problem A @ [c_x, c_y, c_z, r2] = b
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Solve Ax = b
        center = x[:-1]  # First D elements are the center coordinates
        radius = np.sqrt(x[-1] + np.sum(center ** 2))  # Compute radius

        return cls(center, radius)
