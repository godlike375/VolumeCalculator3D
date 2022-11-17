from numpy import ndarray
from model.volume_calculation import rotate, calculate_volume
from model.curve_detection import detect_contours
from model.data_interface import get_files_with_numbers, NumberedImage

class Model:
    def __init__(self, view_model):
        self._view_model = view_model

    def run(self, dir: str) -> tuple[list]:
        images = get_files_with_numbers(dir)
        images.sort(self.extract_number)
        points_3d = []
        for img in images:
            contour_points = detect_contours(img, lower_hsv=[1, 0, 0], upper_hsv=[20, 255, 255],
                                              threshold=254, approximation_rate=0.001)
            points = self.plane_to_3d(contour_points)
            points_3d.append(points)
        volume = calculate_volume(points_3d)
        return points_3d, volume

    def plane_to_3d(self, points, angle, axis=None):
        axis = axis or [0, 1, 0] # axis Y by default
        return [rotate(vector, axis, angle) for vector in points]

    def extract_number(self, image: NumberedImage):
        return image.number
