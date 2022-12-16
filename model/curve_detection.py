import cv2
import numpy as np

UPPER_THRESHOLD = 255


def detect_contours(img, lower_hsv, upper_hsv, threshold, approximation_rate):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(lower_hsv, dtype="uint8")
    upper_hsv = np.array(upper_hsv, dtype="uint8")
    mask = cv2.inRange(image, lower_hsv, upper_hsv)

    _, threshed = cv2.threshold(mask, threshold, UPPER_THRESHOLD, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = sorted(contours, key=cv2.contourArea)[-1]

    # calculate the perimeter of the contour
    arclen = cv2.arcLength(largest_contour, True)
    epsilon = arclen * approximation_rate
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    return approx


def draw_contours(img, points):
    # draw the result
    canvas = img.copy()
    for pt in points:
        cv2.circle(canvas, (pt[0][0], pt[0][1]), 3, (0, 255, 0), -1)

    return canvas

    # cv2.drawContours(canvas, [points], -1, (0,0,255), 1, cv2.LINE_AA)
    # cv2.imshow('kek', canvas)

# img = cv2.imread("test3.jpg")

# arrow_contours = detect_contours(img, lower_hsv=[25, 0, 180], upper_hsv=[45, 255, 255], threshold=254, approximation_rate=0.01)
# цифру распознавать через обрезание по координатам, т.к. она всегда в одном месте
