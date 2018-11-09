import os 
import glob
import shutil
import random

imgdir  = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data/bridge/valid/images/'
annodir = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data/bridge/valid/bridge_anno_val'

IMGDIR = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data/images/rust_image_val'
annolist = sorted(glob.glob('{}/*'.format(annodir)))

for an in annolist:
    basename = os.path.splitext(os.path.basename(an))[0]
    imgname = basename.split('_')[0] + '.jpg'
    imgpaht = os.path.join(IMGDIR,imgname)
    if not os.path.exists(imgpaht):
        continue
    shutil.copy2(imgpaht,imgdir)