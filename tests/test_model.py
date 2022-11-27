from math import radians

import numpy as np
from numpy.testing import assert_array_equal
import cv2

from model.volume_calculation import rotate_vector_by_axis
from model.business_logic import Model
from model.curve_detection import detect_contours


def test_rotate_vector():
    vectors = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    axises = [0, 1, 0], [1, 0, 0], [0, 1, 0]
    results = np.array([0, 0, -1]), np.array([0, 0, 1]), np.array([1, 0, 0])
    for vec, ax, res in zip(vectors, axises, results):
        assert_array_equal(res, rotate_vector_by_axis(vec, ax, radians(90)))

def test_2d_to_3d():
    points = np.array([[1, 0], [0, 1], [1, 1]])
    res = Model.new_rotated_axis(points, radians(90))
    assert_array_equal(res, np.array([[0, 1, 0], [0, 0, 1], [0, 1, 1]]))


def test(num_regression):
    test_img = cv2.imread('curve_detection.jpg')
    contours: np.ndarray = detect_contours(test_img,  lower_hsv=[1, 0, 0], upper_hsv=[22, 255, 255],
                                         threshold=254, approximation_rate=0.00135)
    num_regression.check({'mock_dict': contours.ravel()})
