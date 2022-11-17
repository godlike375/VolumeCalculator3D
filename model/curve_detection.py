import cv2
import numpy as np

def detect_contours(img, lower_hsv, upper_hsv, threshold, approximation_rate):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(lower_hsv, dtype="uint8")
    upper_hsv = np.array(upper_hsv, dtype="uint8")
    mask = cv2.inRange(image, lower_hsv, upper_hsv)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,threshed = cv2.threshold(mask, threshold, 255,cv2.THRESH_BINARY)

    # find contours without approx
    cnts = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2]

    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    # do approx
    epsilon = arclen * approximation_rate
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx

def draw_contours(img, points):
    # draw the result
    canvas = img.copy()
    for pt in points:
        cv2.circle(canvas, (pt[0][0], pt[0][1]), 3, (0,255,0), -1)

    cv2.drawContours(canvas, [points], -1, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('kek', canvas)

#img = cv2.imread("test3.jpg")

#arrow_contours = detect_contours(img, lower_hsv=[25, 0, 180], upper_hsv=[45, 255, 255], threshold=254, approximation_rate=0.01)
# цифру распознавать через обрезание по координатам, т.к. она всегда в одном месте
#draw_contours(img, arrow_contours)

# save
cv2.namedWindow('kek')

cv2.waitKey(0)