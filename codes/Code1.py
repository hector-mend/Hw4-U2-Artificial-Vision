#From the following pictures perform: 
#To the word picture Perfom: 
#A erotion and closing to the letter "P"
#Perform a skeletonization to the letter "R"
#Perform a dilation to the letter "U"
#Detect the 8 corners of the letter "E"
#Perfom a skeletonization to the letter "A" and then do a dilation
#After applying all the previous operations, 
#show the results in one same picture.

import cv2 as cv
import numpy as np

img1 = cv.imread('Prueba.jpg')

p = img1[209:344, 250:345, 2]
r = img1[209:344, 340:415, 2]
u = img1[209:344, 400:505, 2]
e = img1[209:344, 510:595, 2]
e2 = img1[209:344, 510:595]
b = img1[209:344, 590:675, 2]
a = img1[209:344, 675:770, 2]

#P-Letter - Erotion & Closing
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(p, kernel, iterations = 2)
closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)
new_p = closing

#R-Letter - Skeletonization
erosion1 = cv.erode(r, kernel, iterations = 2)
new_r = erosion1

#U-Letter - Dilation
new_u = cv.dilate(u, kernel, iterations=3)

#E-Letter - Corners ero ---> dil
gray = cv.cvtColor(e2, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2,3, 0.04)

#Result is dilated for marking the corners
dst = cv.dilate(dst, None, iterations =3)

#Threshold for an optimal value, it may vary depending on the image.
e2 [dst > 0.1 * dst.max()] = [0, 0, 255]
new_e = e

#B-Letter - No operation
new_b = b

#A-Letter - Skeletonization --> Dilation
erosion3 = cv.erode(a, kernel, iterations = 3)
dil_a = cv.dilate(erosion3, kernel, iterations=4)
new_a =  dil_a

#Dimensions
dimensions = img1.shape
height = img1.shape[0]
width = img1.shape[1]
channels = img1.shape[2] 

#Print Dimensions Results
print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Number of Channels : ', channels) 

#Print Img Results
img_org = cv.hconcat([p, r, u, e, b, a]) 
img_results = cv.hconcat([new_p, new_r, new_u, new_e, new_b, new_a]) 
img_results_fin = cv.vconcat([img_org, img_results])

cv.imshow('Original & Results', img_results_fin)
cv.imwrite("Original & Results2.jpg", img_results_fin)
cv.waitKey(0)
cv.destroyAllWindows()