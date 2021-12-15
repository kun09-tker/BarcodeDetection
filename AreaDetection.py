import cv2
import numpy as np
image = cv2.imread("imgs/images (10).jpg")
imagecopy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC))

blurred = cv2.blur(gradient, (3, 3))

(_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("threshed",cv2.resize(thresh,None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC))


dilate = cv2.dilate(thresh,None,iterations=1)
cv2.imshow("dilete",cv2.resize(dilate,None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC))
thresh = cv2.subtract(dilate,thresh)
cv2.imshow("morphology",cv2.resize(thresh,None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC))

cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

temp_c = sorted(cnts, key = cv2.contourArea, reverse = True)
max = np.max([c.shape[0] for c in temp_c])
for c in temp_c:
    if c.shape[0]==max:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [box], -1, (255,255,255), -1)
        out = cv2.bitwise_not(np.zeros_like(image))
        cv2.imshow("Image", image)
        out[mask == 255] = image[mask == 255]

cv2.imshow("Out", out)
cv2.waitKey(0)
