from math import radians
from pathlib import Path
from unittest.mock import patch, Mock

import cv2
import numpy as np
from numpy.testing import assert_array_equal

from model.business_logic import Model
from model.curve_detection import detect_contours
from model.data_interface import get_files_with_numbers
from model.volume_calculation import rotate_vector_by_axis, calculate_volume


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


def test_contours_detection(num_regression):
    test_img = cv2.imread('curve_detection.jpg')
    contours: np.ndarray = detect_contours(test_img, lower_hsv=[1, 0, 0], upper_hsv=[22, 255, 255],
                                           threshold=254, approximation_rate=0.00135)
    num_regression.check({'mock_dict': contours.ravel()})


def test_data_interface():
    with patch.object(Path, 'iterdir') as mock_iterdir:
        files = []
        for name in 'img_2.jpg', 'img_1.jpg', '2im_3.jpg', 'img_no_num.jpg':
            file = Mock()
            file.name = name
            files.append(file)
        mock_iterdir.return_value = files
        with patch('os.chdir'):
            with patch('cv2.imread'):
                num_images = get_files_with_numbers('')
                nums_correct = [image.number == num for image, num in zip(num_images, (2, 1, 3))]
                if len(nums_correct) > 0:
                    assert all(nums_correct)
                else:
                    assert False


def test_calculate_volume():
    areas = [1, 2, 3, 4, 5]
    assert calculate_volume(areas) == (0.00011018363939899833, 0)

    areas = [0.5, 2.5, 3, 4, 5]
    assert calculate_volume(areas) == (0.00013313856427378965, 1)
