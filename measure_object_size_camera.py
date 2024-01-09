import sys
sys.dont_write_bytecode = True

import cv2
from object_detector import *
import numpy as np

# load aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# load object detector
detector = HomogeneousBgDetector()

# load cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cap.read()

    # get aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:

        # draw polygon arround the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # aruco perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20


        contours = detector.detect_objects(img)

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
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()