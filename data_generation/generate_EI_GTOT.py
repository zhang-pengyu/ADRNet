import os 
from glob import glob
import cv2
from skimage import data,exposure
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import random as rand
import shutil

def gamma_transform(im_array, gamma):
    
    res = exposure.adjust_gamma(im_array,gamma)
    return res

def linear_tranform(im_array,a,b):

    im_new = np.ones((im_array.shape[0],im_array.shape[1],3),dtype=np.uint8)
    for i in range(im_array.shape[0]):
        for j in range(im_array.shape[1]):
            lst = a * im_array[i,j] + b
            im_new[i,j] = [int(ele) if ele < 255 else 255 for ele in lst]
    return im_new

def gamma_transform2(im_array, gamma):

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(im_array, lookUpTable)
    return res      

if __name__ == '__main__':    
    method = 'gamma1'

    data_path = '/media/zpy/Data2/Dataset/GTOT'
    data_dir = glob(os.path.join(data_path,'*'))
    data_dir.sort()
    save_path = '/media/zpy/Data2/Dataset/GTOT_EI/'
    for i in data_dir:
        folder = glob(os.path.join(i,'*'))
        
        t_seq_path = os.path.join(i,'i','*')
        t_seq_dir = glob(t_seq_path)
        t_seq_dir.sort()

        rgb_seq_path = os.path.join(i,'v','*')
        rgb_seq_dir = glob(rgb_seq_path)
        rgb_seq_dir.sort()

        video_name = i.split('/')[-1]
        new_seq_path = save_path + video_name + '/'
        new_t_path = new_seq_path + 'i/'
        new_rgb_path = new_seq_path + 'v/'
        if not os.path.exists(new_seq_path):
            os.mkdir(new_seq_path)
        if not os.path.exists(new_t_path):
            os.mkdir(new_t_path)
        if not os.path.exists(new_rgb_path):
            os.mkdir(new_rgb_path)

        for file in folder:
            file_name = file.split('/')[-1]
            if 'txt' in file_name:
                new_file = new_seq_path + file_name
                shutil.copyfile(file,new_file)
            
        gt_path = i + '/groundTruth_v.txt'
        gt = np.loadtxt(gt_path,delimiter=' ')
        for count, t_frame in enumerate(t_seq_dir):
            bb = gt[count]
            bb = [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]
            rgb_frame = rgb_seq_dir[count]
            if len(bb) == 8:
                xmin = round(min(bb[0],bb[2],bb[4],bb[6]))
                xmax = round(max(bb[0],bb[2],bb[4],bb[6]))
                ymin = round(min(bb[1],bb[3],bb[5],bb[7]))
                ymax = round(max(bb[1],bb[3],bb[5],bb[7]))
                box = np.array([xmin,ymin,xmax-xmin,ymax-ymin],dtype=int)
            else:
                box = np.array(bb,dtype=int)

            t_name = t_frame.split('/')[-1]
            new_t_name = new_t_path + t_name
            im_t = Image.open(t_frame).convert('RGB')
            t_array = np.array(im_t)

            rgb_name = rgb_frame.split('/')[-1]
            new_rgb_name = new_rgb_path + rgb_name
            im_rgb = Image.open(rgb_frame).convert('RGB')
            rgb_array = np.array(im_rgb)

            flag = rand.choices([0,1],[0.2,0.8])
            
            if flag[0] == 1 and count > 0:
                if method == 'gamma':
                    low = 0.1
                    high = 0.7
                    low1 = 1.5 
                    high1 = 4
                    r = 1
                    if count == 0: 
                        gamma0 = rand.uniform(low,high)
                        gamma1 = rand.uniform(low1,high1)
                        gamma = rand.choice([gamma0,gamma1])
                    else:
                        if gamma > low1:
                            diff = rand.uniform(-r,r)
                            gamma = np.clip(gamma+diff,0,high1)
                        elif gamma < high:
                            diff = rand.uniform(-r,r)*0.2
                            gamma = np.clip(gamma+diff,low,high1)
                    res = gamma_transform(im_array,gamma)

                elif method == 'gamma1':
                    low = 0.1
                    high = 0.7
                    low1 = 3 
                    high1 = 5
                    gamma0 = rand.uniform(low,high)
                    gamma1 = rand.uniform(low1,high1)
                    gamma = rand.choice([gamma0,gamma1])
                    
                    res_t = t_array
                    res_rgb = gamma_transform(rgb_array,gamma)

                elif method == 'linear':
                    a = 0.8
                    b = -50
                    # a = rand.uniform(0.1,0.5)
                    # b = rand.uniform(0,255)
                    res = linear_tranform(im_array,a,b)
                
            else:
                res_t = t_array
                res_rgb = rgb_array
            # fig, ax = plt.subplots(1,4)
            # ax[0].imshow(res)
            # ax[1].imshow(res1)
            # ax[2].imshow(res2)
            # ax[3].imshow(res3)
            # plt.show()

            image = Image.fromarray(res_rgb)
            image.save(new_rgb_name,quality=95,subsampling=0)

            image = Image.fromarray(res_t)
            image.save(new_t_name,quality=95,subsampling=0)
