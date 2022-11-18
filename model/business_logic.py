from math import radians

from numpy import ndarray
import numpy
from model.volume_calculation import rotate, calculate_volume
from model.curve_detection import detect_contours
from model.data_interface import get_files_with_numbers, NumberedImage


class Model:
    def __init__(self, view_model):
        self._view_model = view_model

    def run(self, dir: str):
        images = get_files_with_numbers(dir)
        images.sort(key=self.extract_number)
        points_3d = None
        angle = 0
        for num_img in images:
            contour_points = detect_contours(num_img.image, lower_hsv=[1, 0, 0], upper_hsv=[22, 255, 255],
                                              threshold=254, approximation_rate=0.0011)
            centered_points = self.balance_center(num_img.image, contour_points)
            points = self.plane_to_3d(centered_points, radians(angle))
            if points_3d is None:
                points_3d = points
            else:
                points_3d = numpy.concatenate((points_3d, points), axis=0)
            angle += 10
        #volume = calculate_volume(points_3d)
        #self._view_model.set_volume(volume)
        points_3d = list(zip(*points_3d))
        points_3d = numpy.array(points_3d)
        self._view_model.set_points(points_3d)

    def balance_center(self, img, points):
        dx = img.shape[0] // 2
        dy = img.shape[1] // 2
        for point in points:
            point[0][0] = point[0][0] - dx
            point[0][1] = point[0][1] - dy
        return points

    def plane_to_3d(self, points, angle, axis=None):
        axis = axis or [0, 1, 0] # axis Y by default
        points_3d = []
        for vector in points:
            vec_3d = numpy.array([*vector[0], 0])
            points_3d.append(rotate(vec_3d, axis, angle))
        return numpy.array(points_3d)

    def extract_number(self, image: NumberedImage):
        return image.number
