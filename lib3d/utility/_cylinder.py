import numpy as np


class Cylinder:
    def __init__(self, base, axis, radius, height):
        """
        Initialize a Cylinder object.

        Args:
        base (array-like): A point on the base of the cylinder (3D).
        axis (array-like): Direction vector of the cylinder's axis (unit vector).
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        """
        self.base = np.array(base)
        self.axis = np.array(axis) / np.linalg.norm(axis)  # Normalize axis
        self.radius = radius
        self.height = height

    def distances(self, points):
        """
        Compute the shortest distance of points from the cylinder surface.

        Args:
        points (ndarray): (N, 3) array of N points in 3D.

        Returns:
        distances (ndarray): Distance of each point from the cylinder surface.
        """
        points = np.asarray(points)
        v = points - self.base  # Vector from base to points
        t = np.dot(v, self.axis)  # Projection on axis

        # Clamp t within cylinder height range
        t = np.clip(t, 0, self.height)
        proj_v = np.outer(t, self.axis)  # Projected vector
        radial_vec = v - proj_v
        radial_dist = np.linalg.norm(radial_vec, axis=1)

        # Compute distance from surface
        radial_dist_diff = np.abs(radial_dist - self.radius)
        below_base = t == 0
        above_top = t == self.height
        dist_to_base = np.linalg.norm(points - self.base, axis=1)
        dist_to_top = np.linalg.norm(points - (self.base + self.height * self.axis), axis=1)

        return np.where(below_base, dist_to_base, np.where(above_top, dist_to_top, radial_dist_diff))

    def project_points(self, points):
        """
        Projects points onto the closest point on the cylinder surface.

        Args:
        points (ndarray): (N, 3) array of N points in 3D.

        Returns:
        projected_points (ndarray): (N, 3) array of projected points on the cylinder.
        """
        points = np.asarray(points)

        # Vector from base to points
        v = points - self.base  # Shape (N, 3)

        # Project v onto the cylinder axis
        t = np.dot(v, self.axis)  # Shape (N,)
        t = np.clip(t, 0, self.height)  # Clamp to cylinder height

        # Compute radial projection
        proj_v = np.outer(t, self.axis)  # (N,3) projected vectors
        radial_vec = v - proj_v  # Perpendicular vector
        radial_unit = radial_vec / np.linalg.norm(radial_vec, axis=1, keepdims=True)  # Normalize
        radial_projected = self.radius * radial_unit  # Scale to cylinder radius

        # Compute final projected points
        projected_points = self.base + proj_v + radial_projected
        return projected_points
