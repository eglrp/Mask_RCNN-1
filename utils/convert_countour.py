# import the necessary packages
import numpy as np
import cv2
from skimage import morphology
import os 
import glob

# imgname = "/home/buiduchanh/WorkSpace/Javis/Supervised/Mask-RCNN/makedata/result_1.png"
def process_anno(image, basename, dir, Rot):
    # basename = os.path.splitext(os.path.basename(imgname))[0]
    image = cv2.imread(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,thresh_img = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    im2, cnts, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True) # get largest five contour area

    mask_ = np.ones(image.shape[:2], dtype="uint8") * 255

    # if len(cnts) == 0:
    #     return
   
    # if len(cnts) >= 4:
    #     cnts_ = cnts[:3]
    #     cnt1 = cnts[3:]
    #     for c1 in cnt1:
    #         cv2.drawContours(mask_, [c1], -1, 0, -1)
    #     for c_ in cnts_:
    #         if cv2.contourArea(c_) < 360 :
    #             cv2.drawContours(mask_, [c_], -1, 0, -1)
    # else:
    #     for c_ in cnts:
    #         if cv2.contourArea(c_) < 360 :
    #             cv2.drawContours(mask_, [c_], -1, 0, -1)

    # image_ = cv2.bitwise_and(image, image, mask=mask_)
    # Dir = os.path.join(Rot, 'result_3object', dir )
    # if not os.path.exists(Dir):
    #     os.makedirs(Dir)
    # cv2.imwrite('{}/{}.png'.format(Dir, basename),image_)
    
    
    # tmp = []

    for idx, c_ in enumerate(cnts):
        if cv2.contourArea(c_) < 500:
            continue
        if cv2.contourArea(c_) > 500:
            for i in range(len(cnts)):
                if i != idx:
                    cv2.drawContours(mask_, [cnts[i]], -1, 0, -1)        
        
            image_ = cv2.bitwise_and(thresh_img, thresh_img, mask=mask_)
            Dir = os.path.join(Rot, 'result', dir )
            if not os.path.exists(Dir):
                os.makedirs(Dir)
            cv2.imwrite('{}/{}_{}.png'.format(Dir, basename,idx),image_)
            mask_ = np.ones(image.shape[:2], dtype="uint8") * 255
    
   

if __name__ == "__main__":
    
    Root = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data'
    subDir = ['rust_anno_train','rust_anno_val']
    for di in subDir:
        Dir = os.path.join(Root, 'result_colour', di)
        imglist  = sorted(glob.glob('{}/*'.format(Dir)))
        for img in imglist:
            print(img)
            basename = os.path.splitext(os.path.basename(img))[0]
            process_anno(img, basename, di, Root)
