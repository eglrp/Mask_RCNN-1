import cv2 as cv
import numpy as np
import os 
import glob
from convert_countour import process_anno

def readimg(imgname):
    img = cv.imread(imgname)
    # img[0][0] = np.array([255, 255, 255])
    # cv.imwrite('result.png',img)
    for i in range (len(img)):
        for j in range (len(img[i])):
            # print(img[i][j])
            B = img[i][j][0]
            # G = img[i][j][1]
            R = img[i][j][2]
            if B > 200:
                img[i][j] = np.array([0, 0, 0])
            elif R > 200 :
                img[i][j] = np.array([0, 255, 0])
            else :
                img[i][j] = np.array([0, 255, 0])
    
    return img
    # cv.imwrite('read_img/{}.png',img)

if __name__ == "__main__":
    Root = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data'
    subDir = ['rust_anno_train','rust_anno_val']
    for di in subDir:
        Dir = os.path.join(Root, 'annotation', di)
        imglist  = sorted(glob.glob('{}/*'.format(Dir)))
        for img in imglist:
            print(img)
            basename = os.path.splitext(os.path.basename(img))[0]
            basename = basename [:-5]
            img_ = readimg(img) 
            Des = os.path.join(Root,'result_colour',di)
            if not os.path.exists(Des):
                os.makedirs(Des)
            cv.imwrite('{}/{}.png'.format(Des,basename), img_)
            # process_anno(img_, basename, di, Root)