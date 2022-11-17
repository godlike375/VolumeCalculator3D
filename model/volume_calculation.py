
import cv2



from numpy import cross, eye, dot
from scipy.linalg import expm, norm

def rotation_matrix(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

def rotate(vector, axis, degree):
    M0 = rotation_matrix(axis, degree)
    return dot(M0, vector)

def calculate_volume(points_3d):
    raise NotImplementedError

def extract_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, 160, 160, cv2.THRESH_BINARY)

    # find contours without approx
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    # do approx
    eps = 0.00509
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # draw the result
    canvas = img.copy()
    for pt in approx:
        cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)

    return


v, axs, theta = [2, 0, 0], [0, 1, 0], 1.57
print(rotate(v, axs, theta))

# Data for three-dimensional scattered points
