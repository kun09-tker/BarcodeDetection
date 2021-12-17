import numpy as np
import cv2

def AreaDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9, 9))
    gray = clahe.apply(gray)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    # cv2.imshow("dilate", dilate)
    out = None
    cnts, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)

    areaDetect = np.zeros_like(dilate)
    if len(temp_c) != 0:
        temp_c_filter = [temp for temp in temp_c if temp.shape[0] > 10]
        for rect in temp_c_filter:
            if len(temp_c_filter) > 10:
                rect = cv2.minAreaRect(rect)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(areaDetect,[box],-1,(255,255,255),-1)

    areaDetect = cv2.dilate(areaDetect,None,iterations=10)
    # cv2.imshow("dilate", areaDetect)

    cnts, hierarchy = cv2.findContours(areaDetect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    detect_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(detect_c) != 0:
        max = np.max([temp.shape[0] for temp in detect_c])
        for detect in detect_c:
            if detect.shape[0] == max:
                x, y, w, h = cv2.boundingRect(detect)
                rect = cv2.minAreaRect(detect)
                box = np.int0(cv2.boxPoints(rect))
                out = image.copy()[y:y + h, x:x + w]
                cv2.drawContours(image,[box],-1,(0,255,0),2)
    return image,out