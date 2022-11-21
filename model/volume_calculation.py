import cv2

from pyntcloud import PyntCloud
import pandas

from numpy import cross, eye, dot
from scipy.linalg import expm, norm

def rotation_matrix(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

def rotate(vector, axis, degree):
    M0 = rotation_matrix(axis, degree)
    return dot(M0, vector)

def calculate_volume(points_3d):
    pts = pandas.DataFrame(points_3d, columns=['x', 'z', 'y'])
    cloud = PyntCloud(pts)
    id = cloud.add_structure('convex_hull')
    # выпуклая оболочка - внешний контур множества всех точек
    cloud_mesh = cloud.structures[id]
    return cloud_mesh.volume
