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

image = cv2.resize(opened, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Image", image)
cv2.waitKey(0)
