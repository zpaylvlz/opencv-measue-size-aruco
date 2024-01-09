import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

# load aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
#arucoTag = []
#for i in range(4):
    #arucoImg = np.zeros((200, 200), dtype=np.uint8)
    #arucoImg = cv2.aruco.generateImageMarker(aruco_dict, i, 200, arucoImg, 1)
    #arucoTag.append(arucoImg)
    #cv2.imwrite("5x5Tag" + str(i) + ".png", arucoImg)
#aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_5X5_50)

# load object detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# load image
img = cv2.imread("flat.jpg")
height, width = img.shape[:2]
img = cv2.resize(img, (width // 20, height // 20), interpolation=cv2.INTER_AREA)

# get aruco marker
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)


# draw polygon arround the marker
int_corners = np.intp(markerCorners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

# aruco perimeter
aruco_perimeter = cv2.arcLength(markerCorners[0], True)

# pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20

customdetect = HomogeneousBgDetector()
contours = customdetect.detect_objects(img)
img2 = cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
cv2.imwrite("flat_contour.jpg", img2)
# draw objects boundaries
for cnt in contours:

    # get rect
    rect = cv2.minAreaRect(cnt)
    (x, y),(w, h), angle = rect

    # get width and height of the objects by applying the ratio pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio

    # display rectangle
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

#cv2.imshow("Image", img)
cv2.imwrite("flat_result.jpg", img)
cv2.waitKey(0)