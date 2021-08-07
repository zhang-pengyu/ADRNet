import os
import numpy as np
import pickle
from collections import OrderedDict
from glob import glob 



output_path = '../modules/RGBT234.pkl'
set_type = 'RGBT234'
seq_home = '/media/zpy/Data2/Dataset/'+set_type +'/'

seq_dir = glob(os.path.join(seq_home,'*'))

data = {}
for i,seq_path in enumerate(seq_dir):
    if i > 69:
        break
    seqname = seq_path.split('/')[-1]
    print(seqname)
    RGB_img_list = sorted([p for p in os.listdir(seq_home + seqname+'/visible') if os.path.splitext(p)[1] == '.jpg'])
    T_img_list = sorted([p for p in os.listdir(seq_home + seqname+'/infrared') if os.path.splitext(p)[1] == '.jpg'])
    RGB_gt = np.loadtxt(seq_home + seqname + '/visible.txt', delimiter=',')
    T_gt = np.loadtxt(seq_home + seqname + '/infrared.txt', delimiter=',')


    assert len(RGB_img_list) == len(RGB_gt), "Lengths do not match!!"

    data[seqname] = {'RGB_image':RGB_img_list,'T_image':T_img_list, 'RGB_gt':RGB_gt, 'T_gt':T_gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
