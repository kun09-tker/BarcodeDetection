import cv2
import numpy as np
img = cv2.imread("imgs/images (6).jpg")
img_copy = img.copy()
img = cv2.GaussianBlur(img,(1,9),0)

cv2.imshow("Blur",img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])
gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)

cv2.imshow("filter", gray)

(_,edges) = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

cv2.imshow("linesEdges", edges)
lines = cv2.HoughLinesP(edges, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 5)
edges = cv2.dilate(edges,None,iterations=3)
cv2.imshow("mor",edges)
cnts,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

temp_c = sorted(cnts, key = cv2.contourArea, reverse = True)
max = np.max([c.shape[0] for c in temp_c])
for c in temp_c:
    if c.shape[0] != max:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(edges, [box], -1, (0, 255, 0), -1)
        cv2.drawContours(img, [box], -1, (0, 255, 0), -1)

cv2.imshow("edges",edges)
cv2.imshow("linesDetected", img)

cv2.waitKey(0)
cv2.destroyAllWindows()