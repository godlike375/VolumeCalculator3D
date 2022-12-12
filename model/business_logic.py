from math import radians, tan
from itertools import count

import numpy
import numpy as np
from numpy import ndarray

from model.curve_detection import detect_contours
from model.data_interface import get_files_with_numbers, NumberedImage
from model.volume_calculation import rotate_vector_by_axis, calculate_volume, calculate_areas, ABSTRACT_PER_MM3
from common.logger import logger

DEFAULT_SCAN_DEGREE = 180
VOLUME_FROM_CENTER_COEFFICIENT = tan(radians(10))
IMAGES_COUNT = 18
INVALID_RESULT_WARNING = 'Конечный результат может не соответствовать действительности'


class FolderEmptyError(Exception):
    pass


def extract_number(image: NumberedImage):
    return image.number


def add_z_axis(points: ndarray):
    shape = (points.shape[0], points.shape[1] + 1)
    points_3d = np.zeros(shape)
    points_3d[:, [0, 2]] = points
    points_3d[:, 1] = np.zeros_like(points_3d[:, 1])
    return points_3d


def new_rotated_axis(points: ndarray, angle: float, axis=None):
    axis = axis or [0, 0, 1]
    # axis Y by default, [X, Z, Y] - looks like this is the order
    points_3d = add_z_axis(points)
    return [rotate_vector_by_axis(i, axis, angle).tolist() for i in points_3d]


def set_points_center(img: ndarray, points: ndarray):
    dx = img.shape[0] // 2
    dy = img.shape[1] // 2
    points[:, 0] -= dx
    points[:, 1] -= dy
    return points


def image_to_points(image: ndarray, angle: float):
    contour_points = detect_contours(image, lower_hsv=[1, 0, 0], upper_hsv=[64, 255, 255],
                                     threshold=254, approximation_rate=Model.DEFAULT_APPROXIMATION_RATE)
    contour_points = contour_points.reshape((contour_points.shape[0], contour_points.shape[2]))
    centered_points = set_points_center(image, contour_points)
    points_3d = new_rotated_axis(centered_points, radians(angle))
    return centered_points, points_3d


def find_impact_to_volume(object_points: ndarray):
    """
    Возвращает коэффициент влияния площади данного среза на общий объём фигуры.
    Т.к. используется радиальное сканирование, то ближе к центру каждого кадра будет меньше
    расстояние между соседними кадрами, чем отходя дальше от центра.
    Поэтому можно найти центр фигуры, у которой берётся площадь (обычно в центре площадь
    значительно больше, чем по краям) и, умножив на тангентс угла (10 градусов), воспользовавщись
    формулой y=tg(10)*x+0, где y - зазор между сканами, увеличивающийся с расстоянием,
    найти в итоге приблизительный объём фигуры
    """
    most_right = np.max(object_points[:, 0])
    most_left = np.min(object_points[:, 0])
    object_center = (most_right + most_left) // 2
    return abs(object_center) * VOLUME_FROM_CENTER_COEFFICIENT


class Model:
    DEFAULT_APPROXIMATION_RATE = 0.0013
    def __init__(self, view_model):
        self._view_model = view_model

    def get_volume_by_images(self, dir: str):
        try:
            images = get_files_with_numbers(dir)
            self.check_selected_dir_filenames(images)
            images.sort(key=extract_number)
            angle_step = DEFAULT_SCAN_DEGREE / len(images)  # degree
            points_3d = []
            points_2d = []
            for num_img, angle in zip(images, count(0, angle_step)):
                centered_points_2d, points = image_to_points(num_img.image, angle)
                points_2d.append(centered_points_2d)
                points_3d.extend(points)

            areas = np.array(calculate_areas(points_2d))
            dist_from_center_coefficients = np.array([find_impact_to_volume(i) for i in points_2d])
            corrected_areas = areas * dist_from_center_coefficients
            # чем дальше от центра контур гематомы, тем больший вклад в объём он вносит
            # т.к. прямые срезов всё больше расходятся друг от друга с расстоянием
            volume, ignored_gaps = calculate_volume(corrected_areas.tolist())
            if ignored_gaps:
                self._view_model.show_message \
                    ('Предупреждение',
                     f'В процессе обработки изображений на {ignored_gaps} '
                     f'кадрах не удалость распознать обводку кровоизлияния. '
                     f'{INVALID_RESULT_WARNING}')
            self._view_model.set_volume(volume)

            points_3d = numpy.array(points_3d)
            points_3d_unzipped = numpy.array(list(zip(*points_3d)))
            self._view_model.set_points(points_3d_unzipped)
        except FolderEmptyError:
            pass
        except Exception as e:
            self._view_model.show_message('Непредвиденная ошибка', repr(e))
            logger.exception(e)

    def check_selected_dir_filenames(self, images):
        if not len(images):
            self._view_model.show_message('Ошибка', 'Указанная папка пустая')
            raise FolderEmptyError('selected folder is empty')
        numbers = [i.number for i in images]
        unique_numbers = set(numbers)
        if len(numbers) != IMAGES_COUNT:
            self._view_model.show_message('Предупреждение', f'Изображений меньше 18. {INVALID_RESULT_WARNING}')
        if len(numbers) != len(unique_numbers):
            self._view_model.show_message('Предупреждение',
                                          f'Найдены повторяющиеся номера файлов. {INVALID_RESULT_WARNING}')
        if not all([i in range(1, IMAGES_COUNT + 1) for i in numbers]):
            self._view_model.show_message('Предупреждение',
                                          f'Найдены некорректные номера файлов (0>n>18). {INVALID_RESULT_WARNING}')

    @classmethod
    def set_approximation_rate(cls, rate):
        cls.DEFAULT_APPROXIMATION_RATE = rate
