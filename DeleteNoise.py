import numpy as np
import cv2

image = cv2.imread("imgs/images (0).jpg")

cv2.imshow("Input",image)

image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

cv2.imshow("Noise",image)
cv2.waitKey(0)