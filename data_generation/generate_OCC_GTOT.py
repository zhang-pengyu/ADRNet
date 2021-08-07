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

def crop_image(img, bbox, img_size=[107,107], padding=16, is_resize = False, valid=False):
    
    x,y,w,h = np.array(bbox,dtype='float32')
    # img_size = [w,h]

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size[0]
        pad_h = padding * h/img_size[1]
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    if is_resize == 1:
        scaled = np.array(Image.fromarray(cropped).resize((img_size[1],img_size[0])))
    else: 
        scaled = np.array(Image.fromarray(cropped))
    return scaled

if __name__ == '__main__':    
    method = 'gamma1'

    data_path = '/media/zpy/Data2/Dataset/GTOT'
    data_dir = glob(os.path.join(data_path,'*'))
    data_dir.sort()
    save_path = '/media/zpy/Data2/Dataset/GTOT_OCC/'
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
                occ_obj_w = rand.randrange(1,box[2])
                occ_obj_h = rand.randrange(1,box[3])
                x = rand.randrange(box[0],box[0]+box[2])
                y = rand.randrange(box[1],box[1]+box[3])
                                
                crop_patch_t = crop_image(t_array,box,[200,200],50)
                crop_patch_rgb = crop_image(rgb_array,box,[200,200],50)

                t_color = np.mean(crop_patch_t,0)
                rgb_color = np.mean(crop_patch_rgb,0)
                random_color_t = np.mean(t_color,0)
                random_color_rgb = np.mean(rgb_color,0)
                random_color_t = random_color_t.astype(int)
                random_color_rgb = random_color_rgb.astype(int)
                temp = np.array([rand.randrange(-10,10),rand.randrange(-10,10),rand.randrange(-10,10)],dtype=int)
                random_color_rgb = random_color_rgb - temp 
                random_color_t = random_color_t - temp
                occ_obj = np.ones((occ_obj_h,occ_obj_w,3),dtype=np.uint8)
                ## processing RGB
                for i in range(3):
                    occ_obj[:,:,i] = occ_obj[:,:,i] * random_color_rgb[i]
                res = rgb_array
                if res.shape[0] <= y+occ_obj_h:
                    diff = y+occ_obj_h - res.shape[0]
                    occ_obj = occ_obj[0:occ_obj_h-diff,:]

                if res.shape[1] <= x+occ_obj_w:
                    diff = x+occ_obj_w - res.shape[1]
                    occ_obj = occ_obj[:,0:occ_obj_w-diff]

                if res.shape[0] <= y or res.shape[1] <= x:
                    res = rgb_array
                else:
                    res[y:y+occ_obj_h,x:x+occ_obj_w] = occ_obj
                res_rgb = res

                occ_obj = np.ones((occ_obj_h,occ_obj_w,3),dtype=np.uint8)
                ## processing T
                for i in range(3):
                    occ_obj[:,:,i] = occ_obj[:,:,i] * random_color_t[i]
                res = t_array
                if res.shape[0] <= y+occ_obj_h:
                    diff = y+occ_obj_h - res.shape[0]
                    occ_obj = occ_obj[0:occ_obj_h-diff,:]

                if res.shape[1] <= x+occ_obj_w:
                    diff = x+occ_obj_w - res.shape[1]
                    occ_obj = occ_obj[:,0:occ_obj_w-diff]

                if res.shape[0] <= y or res.shape[1] <= x:
                    res = t_array
                else:
                    res[y:y+occ_obj_h,x:x+occ_obj_w] = occ_obj
                res_t = res
            else:
                res_rgb = rgb_array
                res_t = t_array

            image = Image.fromarray(res_rgb)
            image.save(new_rgb_name,quality=95,subsampling=0)

            image = Image.fromarray(res_t)
            image.save(new_t_name,quality=95,subsampling=0)