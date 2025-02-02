# tests/test_cylinder.py
import pytest
import numpy as np
from lib3d.utility._cylinder import Cylinder


def test_cylinder_distance():
    cylinder = Cylinder(base=[0, 0, 0], axis=[0, 0, 1], radius=5)
    points = np.array([[0, 0, 10], [1, 1, 1], [6, 0, 0]])
    distances = cylinder.distances(points)
    assert distances[0] == 5
    assert distances[1] == np.linalg.norm([1, 1, 1])
    assert distances[2] == 1

def test_cylinder_projection():
    cylinder = Cylinder(base=[0, 0, 0], axis=[0, 0, 1], radius=5)
    points = np.array([[0, 0, 10], [1, 1, 1]])
    projected_points = cylinder.project_points(points)
    assert np.allclose(projected_points[0], [0, 0, 0])
    assert np.allclose(projected_points[1], [1, 1, 0])
