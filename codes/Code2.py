#To the fish image
#Extract the fish from the image and fill the whole contourn of the fish. 
#(Region / Hole FIlling)

import cv2 as cv
import numpy as np

#Read image
img2 = cv.imread('Pescado.png')

#Image to grayscale
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#Threshold the input image to binary
ret, th1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

#Invert image
ret, th2 = cv.threshold(th1, 127, 255, cv.THRESH_BINARY_INV)

#Kernel
kernel = np.ones((5,5), np.uint8)

#Find contourns 
blurred = cv.GaussianBlur(gray, (5,5), 0)
cnn1 = cv.Canny(gray, 220, 255)
cnn2 = cv.dilate(cnn1, None, iterations=1)
cnn3 = cv.erode(cnn2, None, iterations=1)
cnts,_ = cv.findContours(cnn3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#Fill image
res = cv.drawContours(th2, cnts, -1, (255, 255, 255), thickness= cv.FILLED)

#Concat images & print results
img_org = cv.hconcat([gray, th1, cnn1]) 
img_org2 = cv.hconcat([cnn2, cnn3, res]) 
img_org3 = cv.vconcat([img_org, img_org2])

cv.imwrite("Final_results.jpg", img_org3)
cv.imshow('Original_Image_Fish', img2)
cv.imshow('Contourn_Filling_Process', img_org3)

cv.waitKey(0)
cv.destroyAllWindows()
