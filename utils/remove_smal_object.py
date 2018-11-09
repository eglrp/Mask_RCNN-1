# import the necessary packages
import numpy as np
import cv2
from skimage import morphology
import os 
import glob

# imgname = ""
def process_anno(image, basename, dir, Rot):
    # basename = os.path.splitext(os.path.basename(imgname))[0]
    image = cv2.imread(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,thresh_img = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    im2, cnts, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True) # get largest five contour area

    mask_ = np.ones(image.shape[:2], dtype="uint8") * 255
    
    for c_ in cnts:
        if cv2.contourArea(c_) < 360 :
            cv2.drawContours(mask_, [c_], -1, 0, -1)

    image_ = cv2.bitwise_and(image, image, mask=mask_)
    Dir = os.path.join(Rot, 'result_3object', dir )
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    cv2.imwrite('{}/{}.png'.format(Dir, basename),image_)
   
    
   

if __name__ == "__main__":
    
    Root = ''
    subDir = ['rust_anno_train','rust_anno_val']
    for di in subDir:
        Dir = os.path.join(Root, 'result_colour', di)
        imglist  = sorted(glob.glob('{}/*'.format(Dir)))
        for img in imglist:
            print(img)
            basename = os.path.splitext(os.path.basename(img))[0]
            process_anno(img, basename, di, Root)
