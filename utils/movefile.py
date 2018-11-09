import os 
import glob
import shutil
import random

imgdir  = ''
annodir = ''

IMGDIR = ''
annolist = sorted(glob.glob('{}/*'.format(annodir)))

for an in annolist:
    basename = os.path.splitext(os.path.basename(an))[0]
    imgname = basename.split('_')[0] + '.jpg'
    imgpaht = os.path.join(IMGDIR,imgname)
    if not os.path.exists(imgpaht):
        continue
    shutil.copy2(imgpaht,imgdir)