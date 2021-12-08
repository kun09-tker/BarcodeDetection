import numpy as np
import cv2
import  BarcodeDetection

image = cv2.imread("imgs/images (10).jpg")
image = BarcodeDetection.SlewRotation(image)
sobel = BarcodeDetection.Sobel(image)

cnts,hierarchy = cv2.findContours(sobel.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

temp_c = sorted(cnts, key = cv2.contourArea, reverse = True)
max = np.max([c.shape[0] for c in temp_c])
for c in temp_c:
    if c.shape[0]/max*100 > 40:
        print(c.shape[0]/max*100)
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)