import numpy as np
import cv2 as cv

img = cv.imread('Prueba.jpg')
e = img[209:344, 510:595]

gray = cv.cvtColor(e, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
e[dst>0.1*dst.max()]=[0,0,255]
cv.imshow('dst', e)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()