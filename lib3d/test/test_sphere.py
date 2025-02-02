# tests/test_sphere.py
import pytest
import numpy as np
from lib3d.utility._sphere import Sphere

def test_sphere_distance():
    sphere = Sphere(center=[0, 0, 0], radius=5)
    points = np.array([[0, 0, 10], [1, 1, 1], [6, 0, 0]])
    distances = sphere.distances(points)
    assert distances[0] == 5
    assert distances[1] == np.linalg.norm([1, 1, 1]) - 5
    assert distances[2] == np.linalg.norm([6, 0, 0]) - 5

def test_sphere_projection():
    sphere = Sphere(center=[0, 0, 0], radius=5)
    points = np.array([[0, 0, 10], [1, 1, 1]])
    projected_points = sphere.project_points(points)
    assert np.allclose(projected_points[0], [0, 0, 5])
    assert np.allclose(projected_points[1], [1.5, 1.5, 0])



