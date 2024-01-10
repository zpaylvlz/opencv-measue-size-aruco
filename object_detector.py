import cv2
import numpy as np
class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sigma = 0.3
        medianValue = np.median(gray)
        lowerThreshold = int(max(0, 1.0 - sigma) * medianValue)
        upperThreshold = int(min(255, 1.0 + sigma) * medianValue)

        # create a mask with adaptive threshold
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.Canny(blurred, lowerThreshold, upperThreshold)
        #mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
        cv2.imwrite("flat_mask.jpg", mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel)
        #cv2.imwrite("flat_mask_dilate.jpg", mask)
        #mask = cv2.erode(mask, kernel)
        
        cv2.imwrite("flat_mask_after.jpg", mask)
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours
    
    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)
    