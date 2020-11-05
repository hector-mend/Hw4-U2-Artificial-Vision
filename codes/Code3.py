#To the "stars" picture:
#Extract the background from the foreground
#Extract the foregound from the background

import cv2 
import numpy as np

# Getting the kernel to be used in Top-Hat
filt =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filt)

# Read image & Change form BGR to RGB & BGR to Gray
input_image = cv2.imread("Cosmos_original.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image2 = cv2.imread("Cosmos_original.jpg")
rgb_img = cv2.cvtColor(input_image2, cv2.COLOR_BGR2RGB) 

#Color ranges
lower_white = (230,230,230)
upper_white = (255,255,255)

#Masks
mask1 = cv2.inRange(rgb_img, lower_white, upper_white)
mask2 = cv2.inRange(input_image2, lower_white, upper_white)
final_mask = mask1 + mask2

#Bitwise and
final_img = cv2.bitwise_and(input_image2, input_image2, mask = final_mask)

#Top-Hat operation
tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)

#Concat images & print results
cv2.imshow("Original", input_image)
cv2.imshow("Tophat", tophat_img)
cv2.imshow("Extraction", final_img)

cv2.imwrite("Top-Hat.jpg", tophat_img)
cv2.imwrite("Extraction.jpg", final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()