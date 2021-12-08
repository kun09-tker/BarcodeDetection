import numpy as np
import cv2

image = cv2.imread("imgs/images (8).jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])
gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
bitwise_not = cv2.bitwise_not(thresh)

cv2.imshow("bitwise",bitwise_not)

cv2.imshow("SD_gray", thresh)

lines = cv2.HoughLinesP(thresh, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(thresh, (x1, y1), (x2, y2), (255, 255, 255), 2)

cv2.imshow("Input", thresh)

thresh = cv2.bitwise_and(bitwise_not,thresh)

cv2.imshow("Output", thresh)
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
print(angle)
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle

(h, w) = image.shape[:2]
center = (w / 2, h / 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.imshow("Rotation",rotated)
# show the output image
cv2.waitKey(0)