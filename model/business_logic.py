from copy import deepcopy
from math import radians
from itertools import count

import numpy
import numpy as np
from numpy import ndarray

from model.curve_detection import detect_contours
from model.data_interface import get_files_with_numbers, NumberedImage
from model.volume_calculation import rotate_vector_by_axis, calculate_volume

DEFAULT_SCAN_DEGREE = 180
DEFAULT_APPROXIMATION_RATE = 0.00135


class Model:
    def __init__(self, view_model):
        self._view_model = view_model

    def run(self, dir: str):
        images = get_files_with_numbers(dir)
        images.sort(key=self.extract_number)
        if not len(images):
            self._view_model.show_message('Ошибка', 'Указанная папка пустая')
            return
        angle_step = DEFAULT_SCAN_DEGREE / len(images)  # degree
        points_3d = []
        points_2d = []
        for num_img, angle in zip(images, count(0, angle_step)):
            centered_points, points = self.image_to_points(num_img.image, angle)
            points_2d.append(centered_points)
            points_3d.extend(points)
        points_3d = numpy.array(points_3d)
        points_3d_unzipped = numpy.array(list(zip(*points_3d)))

        volume, ignored_gaps = calculate_volume(points_2d)
        if ignored_gaps > 0:
            self._view_model.show_message('Предупреждение',
                                          f'В процессе обработки изображений на {ignored_gaps}'
                                          f' кадрах не удалость распознать обводку кровоизлияния, поэтому'
                                          f' результаты работы могут быть менее точными')
        self._view_model.set_volume(volume)
        self._view_model.set_points(points_3d_unzipped)

    @staticmethod
    def image_to_points(image: ndarray, angle: float):
        contour_points = detect_contours(image, lower_hsv=[1, 0, 0], upper_hsv=[64, 255, 255],
                                         threshold=254, approximation_rate=DEFAULT_APPROXIMATION_RATE)
        contour_points = contour_points.reshape((contour_points.shape[0], contour_points.shape[2]))
        centered_points = Model.set_points_center(image, contour_points)
        points_3d = Model.new_rotated_axis(centered_points, radians(angle))
        return centered_points, points_3d

    @staticmethod
    def set_points_center(img: ndarray, points: ndarray):
        dx = img.shape[0] // 2
        dy = img.shape[1] // 2
        points[:, 0] -= dx
        points[:, 1] -= dy
        return points

    @staticmethod
    def new_rotated_axis(points: ndarray, angle: float, axis=None):
        axis = axis or [0, 0, 1]
        # axis Y by default, [X, Z, Y] - looks like this is the order
        shape = (points.shape[0], points.shape[1] + 1)
        points_3d = deepcopy(points)
        points_3d.resize(shape, refcheck=False)
        points_3d[:, [0, 2]] = points
        points_3d[:, 1] = np.zeros_like(points_3d[:, 1])
        return [rotate_vector_by_axis(i, axis, angle).tolist() for i in points_3d]

    def extract_number(self, image: NumberedImage):
        return image.number

    def set_approximation_rate(self, rate):
        global DEFAULT_APPROXIMATION_RATE
        DEFAULT_APPROXIMATION_RATE = rate
