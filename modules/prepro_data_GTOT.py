import os
import numpy as np
import pickle
from collections import OrderedDict
from glob import glob 



# seq_home = '../dataset/'
# seqlist_path = '../vot-otb.txt'
output_path = '../GTOT_TC.pkl'
set_type = 'GTOT_TC'
seq_home = '/media/zpy/Data2/Dataset/'+set_type +'/'

seq_dir = glob(os.path.join(seq_home,'*'))

data = {}
for i,seq_path in enumerate(seq_dir):
    seqname = seq_path.split('/')[-1]
    print(seqname)
    RGB_img_list = sorted([p for p in os.listdir(seq_home + seqname+'/v') if os.path.splitext(p)[1] == '.png'])
    T_img_list = sorted([p for p in os.listdir(seq_home + seqname+'/i') if os.path.splitext(p)[1] == '.png'])
    RGB_gt = np.loadtxt(seq_home + seqname + '/groundTruth_v.txt', delimiter=' ')
    T_gt = np.loadtxt(seq_home + seqname + '/groundTruth_i.txt', delimiter=' ')


    assert len(RGB_img_list) == len(RGB_gt), "Lengths do not match!!"

    
    x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
    y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
    x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
    y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
    RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
    y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
    x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
    y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
    T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    data[seqname] = {'RGB_image':RGB_img_list,'T_image':T_img_list, 'RGB_gt':RGB_gt, 'T_gt':T_gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
