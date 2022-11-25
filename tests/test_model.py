from math import radians
import numpy as np
import pytest

from model.volume_calculation import rotate_vector_by_axis
from model.business_logic import Model


def test_rotate_vector():
    vectors = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    axises = [0, 1, 0], [1, 0, 0], [0, 1, 0]
    results = [0, 0, -1], [0, 0, 1], [1, 0, 0]
    for vec, ax, res in zip(vectors, axises, results):
        assert res == rotate_vector_by_axis(vec, ax, radians(90)).tolist()

def test_2d_to_3d():
    points = np.array([[1, 0], [0, 1], [1, 1]])
    res = Model.new_rotated_axis(points, radians(90))
    assert res.tolist() == [[0, 1, 0], [0, 0, 1], [0, 1, 1]]


