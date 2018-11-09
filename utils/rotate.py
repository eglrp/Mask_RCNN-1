import cv2
import os
import glob

# read image as grey scale
Dir = ''
Des = ''
imglist = sorted(glob.glob('{}/*'.format(Dir)))
for img_ in imglist:
    print(img_)
    basename = os.path.basename(img_)
    img = cv2.imread(img_)
# get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    
    angle90 = 90
    
    scale = 1.0
    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w)) 
    cv2.imwrite('{}/{}'.format(Des,basename),rotated90)
