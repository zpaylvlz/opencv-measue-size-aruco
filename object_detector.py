import cv2
import numpy as np
class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create a mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
        cv2.imwrite("flat_mask.jpg", mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        mask = cv2.dilate(mask, kernel)
        cv2.imwrite("flat_mask_dilate.jpg", mask)
        mask = cv2.erode(mask, kernel)
        
        cv2.imwrite("flat_mask_after.jpg", mask)
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours
    
    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)