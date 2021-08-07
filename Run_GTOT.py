import os
from os.path import join, isdir
from tracker import *
import numpy as np
import argparse
import pickle
import math
import warnings
import time

warnings.simplefilter("ignore", UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
# import the_module_that_warns

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'RGBT234':
        ############################################  have to refine #############################################

        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################

        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

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

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_MDNet():

    ## option setting
    opts['model_path']= './models/ADRNet_GTOT.pth'
    opts['set_type']= 'GTOT'
    opts['visualize'] = False
    
    ## for GTOT
    opts['lr_init'] = 0.00035
    opts['lr_update'] = 0.0002
    opts['lr_mult'] = {'fc6':11}
    opts['maxiter_update'] = 10 
    opts['maxiter_init'] = 65 
    opts['trans_f_expand'] = 1.4

    model_name = opts['model_path'].split('/')[-1]

    ## path initialization
    dataset_path = '/media/zpy/zpy_2T/Tracking_dataset/RGBT_tracking/'


    seq_home = dataset_path + opts['set_type'] 
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    seq_list.sort()
    fps_record = []
    for num,seq in enumerate(seq_list):
        save_path = './results/' + 'ADRNet' +  '/' + seq + '.txt'
        save_folder = './results/' + 'ADRNet'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.exists(save_path):
            continue
        if num<-1:
            continue
        seq_path = seq_home + '/' + seq
        print('——————————Process sequence: '+seq +'——————————————')
        RGB_img_list, T_img_list, RGB_gt, T_gt =genConfig(seq_path,opts['set_type'])
        result, fps = run_mdnet(RGB_img_list, T_img_list, RGB_gt[0], RGB_gt, seq = seq, display=opts['visualize'])
        print ('{} {} , fps:{}'.format(num,seq, fps))
        np.savetxt(save_path,result)
        fps_record.append(fps)
    if len(fps_record):
        print(sum(fps_record)/len(fps_record))


if __name__ =='__main__':

    run_MDNet()
