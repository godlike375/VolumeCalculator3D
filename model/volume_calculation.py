import pandas
from numpy import cross, eye, dot
from pyntcloud import PyntCloud
from scipy.linalg import expm, norm


# 660 pixels = 1.35 mm it equals 489 px / mm

def calc_rotation_matrix(axis, degree):
    return expm(cross(eye(3), axis / norm(axis) * degree))


def rotate_vector_by_axis(vector, axis, degree):
    M0 = calc_rotation_matrix(axis, degree)
    return dot(M0, vector)


def calculate_volume(points_3d):
    points_data = pandas.DataFrame(points_3d, columns=['x', 'y', 'z'])
    cloud = PyntCloud(points_data)

    id = cloud.add_structure('convex_hull')
    # выпуклая оболочка - внешний контур множества всех точек
    cloud_mesh = cloud.structures[id]

    abstract_volume = cloud_mesh.volume
    return abstract_to_real_volume(abstract_volume)


def abstract_to_real_volume(volume):
    print('IMPLEMENT THE CALCULATION')
    return volume
