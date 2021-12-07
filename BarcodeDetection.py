import numpy as np
import cv2

def SlewRotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bitwise_not = cv2.bitwise_not(thresh)
    lines = cv2.HoughLinesP(thresh, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(thresh, (x1, y1), (x2, y2), (255, 255, 255), 1)
    thresh = cv2.bitwise_and(bitwise_not, thresh)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def Sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return opened

def HoughTransform(image):
    img = cv2.GaussianBlur(image, (1, 9), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    (_, edges) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    lines = cv2.HoughLinesP(edges, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(edges, None, iterations=2)
    cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    max = np.max([c.shape[0] for c in temp_c])
    for c in temp_c:
        if c.shape[0] / max * 100 <= 40:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(edges, [box], -1, (0, 255, 0), -1)
            cv2.drawContours(img, [box], -1, (0, 255, 0), -1)
    edges = cv2.bitwise_not(edges)
    return edges

def Combine(image, Sobel,HoughTransform):
    opened = cv2.subtract(Sobel, HoughTransform)
    opened = cv2.erode(opened, None, iterations=2)
    opened = cv2.dilate(opened, None, iterations=7)
    cnts, hierarchy = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    max = np.max([c.shape[0] for c in temp_c])

    for c in temp_c:
        if c.shape[0] / max * 100 >= 40:
            print(c.shape)
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    return image

def BarcodeDetection(path_image):
    image = cv2.imread(path_image)
    cv2.imshow("Input",image)
    image = SlewRotation(image)
    #cv2.imshow("Rotation",image)
    sobel = Sobel(image)
    #cv2.imshow("S",sobel)
    houghtransform = HoughTransform(image)
    #cv2.imshow("H", houghtransform)
    result = Combine(image,sobel,houghtransform)
    return result

result = BarcodeDetection("imgs/images (10).jpg")
cv2.imshow("Output",result)
cv2.waitKey(0)
