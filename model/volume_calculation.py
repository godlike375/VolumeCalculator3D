from statistics import mean

from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import cv2

# 660 pixels = 1.35 mm it equals 489 px / mm

CONTOUR_GAP_DETECTION = 5

def calc_rotation_matrix(axis, degree):
    return expm(cross(eye(3), axis / norm(axis) * degree))


def rotate_vector_by_axis(vector, axis, degree):
    M0 = calc_rotation_matrix(axis, degree)
    return dot(M0, vector)


def calculate_volume(image_points):
    image_areas = [cv2.contourArea(coordinates) for coordinates in image_points]
    ignored_gaps = set()
    for i in image_areas:
        for j in image_areas:
            if max(i, j) / min(i, j) > CONTOUR_GAP_DETECTION:
                ignored_gaps.add(min(i, j))
    for gap in ignored_gaps:
        image_areas.remove(gap)
    return abstract_to_real_volume(mean(image_areas))

def abstract_to_real_volume(volume):
    print('IMPLEMENT THE CALCULATION')
    return volume
