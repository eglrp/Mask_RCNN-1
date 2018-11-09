import os
import glob

Dir = '/home/buiduchanh/WorkSpace/Javis/makedata_MaskRCNN/Mask-RCNN/data/bridge/bridge_anno_val'
filelist = sorted(glob.glob('{}/*'.format(Dir)))
tmp = []
for filename in filelist:
    basename = os.path.splitext(os.path.basename(filename))[0]
    basename = basename.split('_')[0]
    # print(basename)
    if basename in tmp:
        continue
    tmp.append(basename)

for name in tmp:
    file_ = sorted(glob.glob('{}/{}*'.format(Dir,name)))
    for idx,f in enumerate(file_):
        
        basename = os.path.splitext(os.path.basename(f))[0]
        base = basename.split('_')
        newname = base[0] + '_bridge' + '_{}.png'.format(idx)
        newfile = os.path.join(Dir, newname)
        os.rename(f, newfile)
# img = '/home/buiduchanh/WorkSpace/Javis/Supervised/Mask-RCNN/data/result/610033913_rust_52.png'
# im_ = '/home/buiduchanh/WorkSpace/Javis/Supervised/Mask-RCNN/data/result/610033913_rust_5.png'
# os.rename(img,im_)