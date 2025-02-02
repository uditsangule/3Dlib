# tests/test_line.py
import pytest
import numpy as np
from lib3d.utility._line import Line , LineSegment

def test_line_rotation_angles():
    line = Line(vector=[1, 1, 1], point=[0, 0, 0])
    angles, rotation_matrix = line.get_rotation_angles(degrees=True)
    assert np.allclose(angles, [45, -35.26438968, 0])
    assert rotation_matrix.shape == (3, 3)

def test_line_distance_points():
    line = Line(vector=[1, 1, 1], point=[0, 0, 0])
    points = np.array([[0, 0, 10], [1, 1, 1], [5, 5, 5]])
    distances = line.distance_points(points)
    assert np.allclose(distances, [7.07106781, 0, 7.07106781])

def test_line_projection():
    line = Line(vector=[1, 1, 1], point=[0, 0, 0])
    points = np.array([[0, 0, 10], [1, 1, 1]])
    projected_points = line.project_points(points)
    assert np.allclose(projected_points[0], [10, 10, 10])
    assert np.allclose(projected_points[1], [1, 1, 1])


def test_linesegment_projection():
    segment = LineSegment(p1=[1, 1, 1], p2=[4, 5, 6])
    points = np.array([[1, 1, 1], [3, 3, 3], [6, 6, 6]])
    projected_points = segment.project_points(points)
    assert np.allclose(projected_points[0], [1, 1, 1])
    assert np.allclose(projected_points[1], [3, 3, 3])
    assert np.allclose(projected_points[2], [4, 5, 6])

def test_linesegment_distance():
    segment = LineSegment(p1=[1, 1, 1], p2=[4, 5, 6])
    points = np.array([[1, 1, 1], [2, 2, 2], [6, 6, 6]])
    distances = segment.distance_points(points)
    assert np.allclose(distances[0], 0)
    assert np.allclose(distances[1], 0)
    assert np.allclose(distances[2], np.linalg.norm([6, 6, 6]) - 5)
