import cv2
from BarcodeDetection_Camera import AreaDetection
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
name_code = ""
while True:
    ret, frame = cap.read()
    b,r = AreaDetection(frame)
    w,h,c = b.shape
    if r is not None:
        d = decode(r)
        if len(d) > 0:
            name_code = str(d[0].data);
        cv2.putText(b,name_code, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9, 9))
        gray = clahe.apply(gray)
        cv2.imshow("R",gray)
    else:
        name_code = ""
        cv2.putText(b, name_code, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow("B",b)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyWindow()