import numpy as np
import cv2

image = cv2.imread("imgs/images (3).jpg")
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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow("morphology",cv2.resize(opened,None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC))

# HoughTransform

img = cv2.GaussianBlur(imagecopy,(1,9),0)

cv2.imshow("HT_blur",img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])
gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)

cv2.imshow("HT_filter", gray)

(_,edges) = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

cv2.imshow("HT_threshold", edges)

lines = cv2.HoughLinesP(edges, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
edges = cv2.dilate(edges,None,iterations=4)

cv2.imshow("HT_dilate",edges)

cnts,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

temp_c = sorted(cnts, key = cv2.contourArea, reverse = True)
max = np.max([c.shape[0] for c in temp_c])
for c in temp_c:
    if c.shape[0]/max*100 <= 40:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(edges, [box], -1, (0, 255, 0), -1)
        cv2.drawContours(img, [box], -1, (0, 255, 0), -1)

edges = cv2.bitwise_not(edges)

cv2.imshow("HT_HoughLinesP",edges)
cv2.imshow("HT_remove", img)

#  Combine

opened = cv2.subtract(opened,edges)

cv2.imshow("combine", opened)

opened = cv2.erode(opened, None, iterations = 2)
opened = cv2.dilate(opened, None, iterations = 10)

cv2.imshow("dilate", opened)

cnts,hierarchy = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

temp_c = sorted(cnts, key = cv2.contourArea, reverse = True)

max = np.max([c.shape[0] for c in temp_c])

for c in temp_c:
    if c.shape[0]/max*100 > 50:
        print(c.shape)
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Image", image)
cv2.waitKey(0)
