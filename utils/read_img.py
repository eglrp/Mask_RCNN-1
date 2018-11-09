import os
import cv2

img = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data/result_colour/rust_anno_train/610000298.png'

image = cv2.imread(img)
    
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret,thresh_img = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
cv2.imshow('a',thresh_img)
cv2.waitKey()

im2, cnts, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

