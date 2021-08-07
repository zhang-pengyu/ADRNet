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

def avg_blur(im_array, kernel_sz, repeat):
    for i in range(repeat):
        im_array = cv2.blur(im_array , kernel_sz)
    return im_array

def med_blur(im_array,kernel_sz,repeat):
    kernel_sz = kernel_sz[0]
    for i in range(repeat):
        im_array = cv2.medianBlur(im_array , kernel_sz)
    return im_array

def gaussian_blur(im_array,kernel_sz, sigmaX,repeat):
    for i in range(repeat):
        im_array = cv2.GaussianBlur(im_array, kernel_sz, sigmaX)
    return im_array     

def motion_blur(im_array,kernel_sz,repeat):
    kernel_sz = kernel_sz[0]
    kernel = np.eye(kernel_sz,dtype=int)/kernel_sz

    for i in range(repeat):
        im_array = cv2.filter2D(im_array, -1, kernel)
    return im_array   

def motion_blur1(im_array,length,angle):
 
    EPS=np.finfo(float).eps                                 
    alpha = (angle-math.floor(angle/ 180) *180) /180* math.pi
    cosalpha = math.cos(alpha)  
    sinalpha = math.sin(alpha)  
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:  
        xsign = 1
    psfwdt = 1;  
    #模糊核大小
    sx = int(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))  
    sy = int(math.fabs(length*sinalpha + psfwdt - length*EPS))
    sx = max(sx,1)
    sy = max(sy,1)
    psf1=np.zeros((sy,sx))
     
    #psf1是左上角的权值较大，越往右下角权值越小的核。
    #这时运动像是从右下角到左上角移动
    half = length/2
    for i in range(0,sy):
        for j in range(0,sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j) 
            if  rad >= half and math.fabs(psf1[i][j]) <= psfwdt:  
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)  
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);  
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    #运动方向是往左上运动，锚点在（0，0）
    anchor=(0,0)
    #运动方向是往右上角移动，锚点一个在右上角
    #同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle<90 and angle>0:
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,0)
    elif angle<-90 :#同理：往右下角移动
        psf1=np.flipud(psf1)
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
    elif angle>-90 and angle<0:#同理：往左下角移动
        psf1=np.flipud(psf1)
        anchor=(0,psf1.shape[0]-1)
    psf1=psf1/psf1.sum()
    res = cv2.filter2D(im_array,-1,psf1,anchor=anchor)
    return res

if __name__ == '__main__':    
    method = 'gamma1'

    data_path = '/media/zpy/Data2/Dataset/GTOT'
    data_dir = glob(os.path.join(data_path,'*'))
    data_dir.sort()
    save_path = '/media/zpy/Data2/Dataset/GTOT_TC/'
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
                kernel_h = rand.randrange(1,10,2)
                kernel_w = rand.randrange(1,10,2)
                kernel_sz = (kernel_h,kernel_w)
                repeat = rand.randrange(5,15)
                res_t = avg_blur(t_array, kernel_sz, repeat)
                res_rgb = rgb_array
            # res1 = med_blur(im_array, kernel_sz, repeat)
            # res2 = gaussian_blur(im_array, kernel_sz, sigmaX, repeat)
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
