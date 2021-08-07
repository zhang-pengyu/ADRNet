import sys
import numpy as np
import PIL
from PIL import Image


import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import *

import matplotlib.patches as patches

import os
from sample_generator import *

import sys
from pretrain_options import *

from img_cropper import *



class RegionDataset(data.Dataset):
    def __init__(self, img_dir, RGB_img_list, T_img_list, RGB_gt, T_gt, receptive_field, opts):
        dataset_name = img_dir.split('/')[-2]
        if dataset_name == 'RGBT234':
            self.RGB_img_list = np.array([os.path.join(img_dir,'visible',img) for img in RGB_img_list])
            self.T_img_list = np.array([os.path.join(img_dir,'infrared',img) for img in T_img_list])
        elif dataset_name == 'GTOT':
            self.RGB_img_list = np.array([os.path.join(img_dir,'v',img) for img in RGB_img_list])
            self.T_img_list = np.array([os.path.join(img_dir,'i',img) for img in T_img_list])
        
        self.RGB_gt = RGB_gt
        self.T_gt = T_gt

        self.batch_frames = pretrain_opts['batch_frames']
        self.batch_pos = pretrain_opts['batch_pos']
        self.batch_neg = pretrain_opts['batch_neg']

        self.overlap_pos = pretrain_opts['overlap_pos']
        self.overlap_neg = pretrain_opts['overlap_neg']


        self.crop_size = pretrain_opts['img_size']
        self.padding = pretrain_opts['padding']

        self.index = np.random.permutation(len(self.RGB_img_list))
        self.pointer = 0

        image = Image.open(self.RGB_img_list[0]).convert('RGB')
        self.scene_generator = SampleGenerator('gaussian', image.size,trans_f=1.5, scale_f=1.2,valid=True)
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

        self.receptive_field = receptive_field

        self.interval = pretrain_opts['frame_interval']
        self.img_crop_model = imgCropper(pretrain_opts['padded_img_size'])
        self.img_crop_model.eval()
        if pretrain_opts['use_gpu']:
            self.img_crop_model.gpuEnable()

    def __iter__(self):
        return self

    def __next__(self):

        next_pointer = min(self.pointer + self.batch_frames, len(self.RGB_img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.RGB_img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer


        n_pos = self.batch_pos
        n_neg = self.batch_neg

        RGB_scenes = []
        T_scenes = []
        init_RGB_targets = []
        init_T_targets = []
        for i, (RGB_img_path,T_img_path,RGB_bbox,T_bbox) in enumerate(zip(self.RGB_img_list[idx], self.T_img_list[idx], self.RGB_gt[idx], self.T_gt[idx])):
            RGB_image = Image.open(RGB_img_path).convert('RGB')
            RGB_image = np.asarray(RGB_image)
            T_image = Image.open(T_img_path).convert('RGB')
            T_image = np.asarray(T_image)

            ## get initial target
            init_RGB_image = Image.open(self.RGB_img_list[0]).convert('RGB')
            init_RGB_image = np.asarray(init_RGB_image)
            init_T_image = Image.open(self.T_img_list[0]).convert('RGB')
            init_T_image = np.asarray(init_T_image)
            init_RGB_bbox = self.RGB_gt[0]
            init_T_bbox = self.T_gt[0]

            init_RGB_target = init_RGB_image[int(init_RGB_bbox[1]):int(init_RGB_bbox[1]+init_RGB_bbox[3]),int(init_RGB_bbox[0]):int(init_RGB_bbox[0]+init_RGB_bbox[2]),:]
            init_T_target = init_T_image[int(RGB_bbox[1]):int(RGB_bbox[1]+RGB_bbox[3]),int(RGB_bbox[0]):int(RGB_bbox[0]+RGB_bbox[2]),:]
            
            init_RGB_target = np.asarray(Image.fromarray(init_RGB_target).resize((95,95),Image.BILINEAR))
            init_T_target = np.asarray(Image.fromarray(init_T_target).resize((95,95),Image.BILINEAR))
            init_RGB_target = init_RGB_target[np.newaxis,:,:,:]
            init_T_target = init_T_target[np.newaxis,:,:,:]
            init_RGB_target = init_RGB_target.transpose(0,3,1,2)
            init_T_target = init_T_target.transpose(0,3,1,2)

            init_RGB_target = Variable(torch.from_numpy(init_RGB_target).float()).cuda()
            init_T_target = Variable(torch.from_numpy(init_T_target).float()).cuda()

            ishape = RGB_image.shape
            pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1],ishape[0]), 0.1, 1.2, 1.1, False), RGB_bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1],ishape[0]), 1, 1.2, 1.1, False), RGB_bbox, n_neg, overlap_range=self.overlap_neg)
            
            # compute padded sample
            padded_x1 = (neg_examples[:, 0]-neg_examples[:,2]*(pretrain_opts['padding']-1.)/2.).min()
            padded_y1 = (neg_examples[:, 1]-neg_examples[:,3]*(pretrain_opts['padding']-1.)/2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2]*(pretrain_opts['padding']+1.)/2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3]*(pretrain_opts['padding']+1.)/2.).max()
            padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

            jitter_scale = 1.1 ** np.clip(3.*np.random.randn(1,1),-2,2)
            crop_img_size = (padded_scene_box[2:4] * ((pretrain_opts['img_size'], pretrain_opts['img_size']) / RGB_bbox[2:4])).astype('int64') * jitter_scale[0][0]
            RGB_cropped_image, _ = self.img_crop_model.crop_image(RGB_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            RGB_cropped_image = RGB_cropped_image - 128.

            T_cropped_image, _ = self.img_crop_model.crop_image(T_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            T_cropped_image = T_cropped_image - 128.


            if pretrain_opts['use_gpu']:
                RGB_cropped_image = RGB_cropped_image.data.cpu()
                T_cropped_image = T_cropped_image.data.cpu()

                init_RGB_target = init_RGB_target.cpu()
                init_T_target = init_T_target.cpu()

            RGB_scenes.append(RGB_cropped_image)
            T_scenes.append(T_cropped_image)

            init_RGB_targets.append(init_RGB_target)
            init_T_targets.append(init_T_target)
            ## get current frame and heatmap

            rel_bbox = np.copy(RGB_bbox)
            rel_bbox[0:2] -= padded_scene_box[0:2]

            jittered_obj_size = jitter_scale[0][0]*float(pretrain_opts['img_size'])

            batch_num = np.zeros((pos_examples.shape[0], 1))
            pos_rois = np.copy(pos_examples)
            pos_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), pos_rois.shape[0], axis=0)
            pos_rois = samples2maskroi(pos_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),RGB_bbox[2:4], pretrain_opts['padding'])
            pos_rois = np.concatenate((batch_num, pos_rois), axis=1)

            batch_num = np.zeros((neg_examples.shape[0], 1))
            neg_rois = np.copy(neg_examples)
            neg_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), neg_rois.shape[0], axis=0)
            neg_rois = samples2maskroi(neg_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),RGB_bbox[2:4], pretrain_opts['padding'])
            neg_rois = np.concatenate((batch_num, neg_rois), axis=1)

            if i==0:
                total_pos_rois = [torch.from_numpy(np.copy(pos_rois).astype('float32'))]
                total_neg_rois = [torch.from_numpy(np.copy(neg_rois).astype('float32'))]
            else:
                total_pos_rois.append(torch.from_numpy(np.copy(pos_rois).astype('float32')))
                total_neg_rois.append(torch.from_numpy(np.copy(neg_rois).astype('float32')))

        return RGB_scenes, T_scenes, total_pos_rois, total_neg_rois, init_RGB_targets, init_T_targets

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0,3,1,2).astype('float32')
        regions = regions - 128.
        return regions
