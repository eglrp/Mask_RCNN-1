import os
import glob
import shutil
IMG = ''
ANNO = ''
Dir = ''
for dir1 in os.listdir(Dir):
    subdir1 = os.path.join(Dir,dir1)
    for dir2 in os.listdir(subdir1):
        subdir2 = os.path.join(subdir1,dir2)
        for dir3 in os.listdir(subdir2):
            subdir3 = os.path.join(subdir2,dir3)
            imglist = sorted(glob.glob('{}/*'.format(subdir3)))
            for idx,img in enumerate(imglist):
                if idx %2 != 0 :
                    print(img)
                    shutil.copy2(img,ANNO)
                else:
                    shutil.copy2(img,IMG)
                    