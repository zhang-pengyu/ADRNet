import os
import scipy.io
import numpy as np
from collections import OrderedDict
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time

import sys
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class Ensemble_operation(nn.Module):
    
    def __init__(self, input_size):
        super(Ensemble_operation, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_size)*1e-2, requires_grad=True)

    def forward(self, x):
        output = F.softmax(self.weight, 0)[0] * x[:,0,:,:] + F.softmax(self.weight, 0)[1] * x[:,1,:,:]
        return output.unsqueeze(1)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None,K=1 ,train = True):
        super(MDNet, self).__init__()
        self.K = K
        self.RGB_layers = nn.Sequential(OrderedDict([
                ('conv1_RGB', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('conv2_RGB', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1),
                                        nn.ReLU(),
                                        LRN(),
                                        )),
                ('conv3_RGB', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3),
                                        nn.ReLU(),
                                        ))
        ]))

        self.T_layers = nn.Sequential(OrderedDict([
                ('conv1_T', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('conv2_T', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1),
                                        nn.ReLU(),
                                        LRN(),
                                        )),
                ('conv3_T', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3),
                                        nn.ReLU(),
                                        ))
        ]))

        self.temp = nn.Sequential(OrderedDict([
                ('temp', nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        ))
        ])) 

        self.linear_layers = nn.Sequential(OrderedDict([                 
                ('fc4',   nn.Sequential(
                                        nn.Linear(1024 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        self.S_attention = nn.Sequential(OrderedDict([
                ('S_att_res_conv1', nn.Sequential( 
                                        nn.InstanceNorm2d(1024),
                                        nn.Conv2d(1024, 256, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('S_att_res_conv2', nn.Sequential( 
                                        nn.Conv2d(256, 64, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('S_att_res_deconv1', nn.Sequential( 
                                        nn.ConvTranspose2d(64, 8, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('S_att_res_deconv2', nn.Sequential( 
                                        nn.ConvTranspose2d(8, 1,  kernel_size=3, stride=1),
                                        
                                        )),
                ('S_att_en_conv1', nn.Sequential( 
                                        nn.InstanceNorm2d(2),
                                        Ensemble_operation(2),
                                        nn.Sigmoid(),
                                        )),
        ])) 

        self.C_attention = nn.Sequential(OrderedDict([
                ('C_att_fc1',   nn.Sequential(
                                        nn.Linear(1024, 128),
                                        nn.ReLU())),
                ('C_att_fc2_none',   nn.Sequential(
                                        nn.Linear(128, 1024),
                                        )),
                ('C_att_fc2_EI',   nn.Sequential(
                                        nn.Linear(128, 1024),
                                        )),
                ('C_att_fc2_OCC',   nn.Sequential(
                                        nn.Linear(128, 1024),
                                        )),
                ('C_att_fc2_TC',   nn.Sequential(
                                        nn.Linear(128, 1024),
                                        )),
                ('C_att_fc2_MB',   nn.Sequential(
                                        nn.Linear(128, 1024),
                                        )),
                            
        ])) 


        self.residual_layers = nn.Sequential(OrderedDict([

                ('conv4_none',          nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv4_EI',            nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv4_OCC',           nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv4_TC',            nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv4_MB',            nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        )),                                
        ])) 

        self.roi_align_model = PrRoIPool2D(3, 3, 1./8)
        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers. feat_map is bottom layer of ROI_align_layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        if model_path is not None:
            if train:
                if model_path.split('/')[-1] == 'rt-mdnet.pth':
                    self.load_model_part(model_path)
                elif 'epoch' in model_path.split('/')[-1]:
                    self.load_model_exclude_residual(model_path)
                elif os.path.splitext(model_path)[1] == '.mat':
                    self.load_mat_model(model_path)
                elif os.path.isdir(model_path):
                    self.load_models(model_path)
                elif 'fc' in model_path.split('/')[-1]:
                    self.load_model_tracking(model_path)
                elif 'CANet' in model_path.split('/')[-1]:
                    self.load_model_exclude_SANet(model_path)
                elif 'SANet' in model_path.split('/')[-1]:
                    self.load_model_exclude_CANet(model_path)
                elif 'AGRB' in model_path.split('/')[-1]:
                    self.load_model_exclude_SCANet(model_path)
                else:
                    raise RuntimeError("Unkown model format: %s" % (model_path))
            else:
                self.load_model_tracking(model_path)
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.RGB_layers.named_children():
            append_params(self.params, module, name)
        for name, module in self.T_layers.named_children():
            append_params(self.params, module, name)
        for name, module in self.residual_layers.named_children():
            append_params(self.params, module, name)
        for name, module in self.linear_layers.named_children():
            append_params(self.params, module, name)
        for name, module in self.S_attention.named_children():
            append_params(self.params, module, name)
        for name, module in self.C_attention.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def corr_fun(self, Kernel_tmp, Feature, KERs=None):
        size = Kernel_tmp.size()
        if len(Feature) == 1:
            Kernel = Kernel_tmp.view(size[1], size[2] * size[3]).transpose(0, 1)
            Kernel = Kernel.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                Kernel = KERs[0]
            corr = torch.nn.functional.conv2d(Feature, Kernel.contiguous())
        else:
            CORR = []
            Kernel = []
            Kernel = Kernel_tmp.view(size[1], size[2] * size[3]).transpose(0, 1)
            Kernel = Kernel.unsqueeze(2).unsqueeze(3)
            for i in range(len(Feature)):
                fea = Feature[i:i + 1]
                if not (type(KERs) == type(None)):
                    Kernel = KERs[0]
                co = F.conv2d(fea, Kernel.contiguous())
                CORR.append(co)
            corr = torch.cat(CORR, 0)
        return corr

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def get_target_feat(self, RGB_target, T_target):

        RGB_target = self.RGB_layers(RGB_target)
        T_target = self.T_layers(T_target)
        self.feat_target_RGBT = torch.cat([RGB_target, T_target], 1)

    def forward(self, RGB_x, T_x, k=0, in_layer='conv1', out_layer='fc6'):

        if in_layer == 'conv1':

            feat_RGBT = torch.cat([self.RGB_layers(RGB_x), self.T_layers(T_x)], 1)

            residual_none = self.residual_layers.conv4_none(feat_RGBT)
            residual_EI = self.residual_layers.conv4_EI(feat_RGBT)
            residual_MB = self.residual_layers.conv4_MB(feat_RGBT)
            residual_OCC = self.residual_layers.conv4_OCC(feat_RGBT)
            residual_TC = self.residual_layers.conv4_TC(feat_RGBT)
            
            feat_sum  = residual_none + residual_EI + residual_MB + residual_OCC + residual_TC
            num_group = feat_sum.size()[0]

            '''spatial attention'''
            spatial_attention_main = torch.max(self.corr_fun(self.feat_target_RGBT,feat_RGBT), dim=1, keepdim=True)[0]

            spatial_attention_res = self.S_attention.S_att_res_deconv2(self.S_attention.S_att_res_deconv1(self.S_attention.S_att_res_conv2(self.S_attention.S_att_res_conv1(feat_sum))))            
            
            spatial_attention = torch.cat([spatial_attention_main, spatial_attention_res],1)
            spatial_attention = self.S_attention.S_att_en_conv1(spatial_attention)

            """channel attention"""
            feat_GAP = self.C_attention.C_att_fc1(self.GAP(feat_sum).view(-1,feat_sum.size()[1]))

            feat_GAP_none = self.C_attention.C_att_fc2_none(feat_GAP)
            feat_GAP_EI = self.C_attention.C_att_fc2_EI(feat_GAP)
            feat_GAP_OCC = self.C_attention.C_att_fc2_OCC(feat_GAP)
            feat_GAP_MB = self.C_attention.C_att_fc2_MB(feat_GAP)
            feat_GAP_TC = self.C_attention.C_att_fc2_TC(feat_GAP)
            
            if num_group == 1:
                channel_attention = nn.functional.softmax(torch.cat([feat_GAP_none, feat_GAP_EI,feat_GAP_MB,feat_GAP_OCC,feat_GAP_TC], 0),dim=0)
                
                feat_C = residual_none * channel_attention[0,:].view(1,channel_attention.size()[1],1,1) + \
                residual_EI * channel_attention[1,:].view(1,channel_attention.size()[1],1,1) + \
                residual_MB * channel_attention[2,:].view(1,channel_attention.size()[1],1,1) + \
                residual_OCC * channel_attention[3,:].view(1,channel_attention.size()[1],1,1) + \
                residual_TC * channel_attention[4,:].view(1,channel_attention.size()[1],1,1)
            else:
                channel_attention = nn.functional.softmax(torch.cat([feat_GAP_none.unsqueeze(1), feat_GAP_EI.unsqueeze(1),feat_GAP_MB.unsqueeze(1),feat_GAP_OCC.unsqueeze(1),feat_GAP_TC.unsqueeze(1)], 1),dim=1)
                
                feat_C = residual_none * channel_attention[:,0,:].view(num_group,channel_attention.size()[2],1,1) + \
                residual_EI * channel_attention[:,1,:].view(num_group,channel_attention.size()[2],1,1) + \
                residual_MB * channel_attention[:,2,:].view(num_group,channel_attention.size()[2],1,1) + \
                residual_OCC * channel_attention[:,3,:].view(num_group,channel_attention.size()[2],1,1) + \
                residual_TC * channel_attention[:,4,:].view(num_group,channel_attention.size()[2],1,1)
                    
            feat = feat_C * spatial_attention + feat_RGBT

        else:
            x = self.linear_layers(RGB_x)
            x = self.branches[k](x)
            if out_layer=='fc6':
                return x
            elif out_layer=='fc6_softmax':
                return F.softmax(x)  

        if out_layer == 'conv4':
            return feat
        else:
            x = self.linear_layers(feat)
            x = self.branches[k](x)
            if out_layer=='fc6':
                return x
            elif out_layer=='fc6_softmax':
                return F.softmax(x)   
        

    def load_model_part(self, model_path):
        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            layer_name = k.split('.')[0] + '_RGB.'+k.split('.')[1] + '.'+k.split('.')[2]
            if 'RGB_layers.'+ layer_name in model_dict.keys():
                state_dict.update({'RGB_layers.'+ layer_name:v})
            layer_name = k.split('.')[0] + '_T.'+k.split('.')[1] + '.'+k.split('.')[2]
            if 'T_layers.'+ layer_name in model_dict.keys():
                state_dict.update({'T_layers.'+ layer_name:v})
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_models(self, model_path):
        model_dir = glob(os.path.join(model_path,'*'))
        model_dict = self.state_dict()
        for i in model_dir:
            attr = i.split('/')[-1].split('.')[0]
            states = torch.load(i)
            states = states['shared_layers']
            state_dict = {}
            if attr == 'None':
                for k,v in states.items():
                    if 'RGB_layers.'+ k in model_dict.keys():
                        state_dict.update({'RGB_layers.'+ k:v})
                    if 'T_layers.'+ k in model_dict.keys():
                        state_dict.update({'T_layers.'+ k:v})
                    if 'conv4' in k and 'weight' in k:
                        state_dict.update({'residual_layers.conv4_none.1.weight':v})
                    if 'conv4' in k and 'bias' in k:
                        state_dict.update({'residual_layers.conv4_none.1.bias':v})
                    if 'linear_layers.'+ k in model_dict.keys():
                        state_dict.update({'linear_layers.'+ k:v})
                model_dict.update(state_dict)
            elif attr == 'EI':
                for k,v in states.items():
                    if 'conv4' in k and 'weight' in k:
                        state_dict.update({'residual_layers.conv4_EI.1.weight':v})
                    if 'conv4' in k and 'bias' in k:
                        state_dict.update({'residual_layers.conv4_EI.1.bias':v})
                model_dict.update(state_dict)
            elif attr == 'OCC':
                for k,v in states.items():
                    if 'conv4' in k and 'weight' in k:
                        state_dict.update({'residual_layers.conv4_OCC.1.weight':v})
                    if 'conv4' in k and 'bias' in k:
                        state_dict.update({'residual_layers.conv4_OCC.1.bias':v})
                model_dict.update(state_dict)  
            elif attr == 'TC':
                for k,v in states.items():
                    if 'conv4' in k and 'weight' in k:
                        state_dict.update({'residual_layers.conv4_TC.1.weight':v})
                    if 'conv4' in k and 'bias' in k:
                        state_dict.update({'residual_layers.conv4_TC.1.bias':v}) 
                model_dict.update(state_dict)
            elif attr == 'MB':    
                for k,v in states.items():
                    if 'conv4' in k and 'weight' in k:
                        state_dict.update({'residual_layers.conv4_MB.1.weight':v})
                    if 'conv4' in k and 'bias' in k:
                        state_dict.update({'residual_layers.conv4_MB.1.bias':v})
                model_dict.update(state_dict)
        self.load_state_dict(model_dict)
            
                

    def load_model_tracking(self, model_path):
        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'residual_layers.'+ k in model_dict.keys():
                state_dict.update({'residual_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
            if 'S_attention.'+ k in model_dict.keys():
                state_dict.update({'S_attention.'+ k:v})
            if 'C_attention.'+ k in model_dict.keys():
                state_dict.update({'C_attention.'+ k:v})

        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_model_exclude_SCANet(self, model_path):
    
        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'residual_layers.'+ k in model_dict.keys():
                state_dict.update({'residual_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_model_exclude_SANet(self, model_path):

        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'residual_layers.'+ k in model_dict.keys():
                state_dict.update({'residual_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
            if 'C_attention.'+ k in model_dict.keys():
                state_dict.update({'C_attention.'+ k:v})
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_model_exclude_CANet(self, model_path):

        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'residual_layers.'+ k in model_dict.keys():
                state_dict.update({'residual_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
            if 'S_attention.'+ k in model_dict.keys():
                state_dict.update({'S_attention.'+ k:v})
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)


    def load_model_seperate(self, model_path):
        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'residual_layers.'+ k in model_dict.keys():
                state_dict.update({'residual_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
            if 'attrNet.'+ k in model_dict.keys():
                state_dict.update({'attrNet.'+ k:v})

        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_model_exclude_residual(self, model_path):
        states = torch.load(model_path)
        states = states['shared_layers']
        model_dict = self.state_dict()
        state_dict = {}
        for k,v in states.items():
            if 'RGB_layers.'+ k in model_dict.keys():
                state_dict.update({'RGB_layers.'+ k:v})
            if 'T_layers.'+ k in model_dict.keys():
                state_dict.update({'T_layers.'+ k:v})
            if 'linear_layers.'+ k in model_dict.keys():
                state_dict.update({'linear_layers.'+ k:v})
        
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    def trainSpatialTransform(self, image, bb):

        return


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]

        loss = (pos_loss.sum() + neg_loss.sum())/(pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        return prec.item()



