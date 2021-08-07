import os
from os.path import join, isdir
from tracker import *
import numpy as np
import argparse
import pickle
import math
import warnings
import time
from glob import glob

warnings.simplefilter("ignore", UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
# import the_module_that_warns

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)

def run_MDNet():

    ## option setting
    opts['model_path']= './models/ADRNet_RGBT234.pth'
    opts['visualize'] = True
    
    ## for RGBT234
    opts['lr_init'] = 0.0003
    opts['lr_update'] = 0.0003
    opts['lr_mult'] = {'fc6':10}
    opts['maxiter_update'] = 15 
    opts['maxiter_init'] = 50 

    model_name = opts['model_path'].split('/')[-1]

    rgb_path = './demo/visible/'
    t_path = './demo/infrared/'

    rgb_dir = glob(rgb_path+'*.jpg')
    t_dir = glob(t_path+'*.jpg')

    rgb_dir.sort()
    t_dir.sort()

    gt = np.loadtxt('./demo/visible.txt',delimiter = ',')

    result, fps = run_mdnet(rgb_dir, t_dir, gt[0], gt, seq = [], display=opts['visualize'])
    print ('fps:{}'.format(fps))


if __name__ =='__main__':
        
    run_MDNet()