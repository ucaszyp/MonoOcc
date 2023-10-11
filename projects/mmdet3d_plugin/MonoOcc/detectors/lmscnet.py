# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MonoOcc/blob/main/LICENSE


import os
import seaborn as sns
import matplotlib.pylab as plt

from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.MonoOcc.utils.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss, BCE_ssc_loss

@DETECTORS.register_module()
class LMSCNet_SS(MVXTwoStageDetector):
    def __init__(self,
                 class_num=None,
                 input_dimensions=None,
                 out_scale=None,
                 gamma=0,
                 alpha=0.5,
                 dataset='nuscenes',
                 uncertain=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):

        super(LMSCNet_SS,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        self.uncertain = uncertain
        self.dataset = dataset
        self.out_scale=out_scale
        self.nbr_classes = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
        f = self.input_dimensions[1]

        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])

        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level1 + 0.001)
        )

        self.Encoder_block1 = nn.Sequential(
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block2 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block3 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block4 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
        self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

        # Treatment output 1:4
        if self.out_scale=="1_4" or self.out_scale=="1_2" or self.out_scale=="1_1":
          self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
          self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
          self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
          self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

        # Treatment output 1:2
        if self.out_scale=="1_2" or self.out_scale=="1_1":
          self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
          self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
          self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

        # Treatment output 1:1
        if self.out_scale=="1_1":
          self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
          self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
        
        if self.uncertain:
          if self.out_scale=="1_1":
            self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
          elif self.out_scale=="1_2":
            self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
            # self.upconv4 = UpConv3D(3, 3)
            # self.skip_conv4 = SkipConv(6, 3)
            # self.upconv3 = UpConv3D(3, 3)
            # self.skip_conv3 = SkipConv(6, 3)
            # self.upconv2 = UpConv3D(3, 3)
            # self.skip_conv2 = SkipConv(6, 3)
          elif self.out_scale=="1_4":
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
          elif self.out_scale=="1_8":
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes + 1, [1, 2, 3])
            
        else:
          if self.out_scale=="1_1":
            self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
          elif self.out_scale=="1_2":
            self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
            self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
          elif self.out_scale=="1_4":
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
          elif self.out_scale=="1_8":
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])      

    def step(self, input):

        # input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
        # input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # print(input.shape) [4, 32, 256, 256]

        # Encoder block
        _skip_1_1 = self.Encoder_block1(input)
        # print('_skip_1_1.shape', _skip_1_1.shape)  # [1, 32, 256, 256]
        _skip_1_2 = self.Encoder_block2(_skip_1_1)
        # print('_skip_1_2.shape', _skip_1_2.shape)  # [1, 48, 128, 128]
        _skip_1_4 = self.Encoder_block3(_skip_1_2) 
        # print('_skip_1_4.shape', _skip_1_4.shape)  # [1, 64, 64, 64]
        _skip_1_8 = self.Encoder_block4(_skip_1_4) 
        # print('_skip_1_8.shape', _skip_1_8.shape)  # [1, 80, 32, 32]

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

        # print('out_scale_1_8__2D.shape', out_scale_1_8__2D.shape)  # [1, 4, 32, 32]

        if self.out_scale=="1_8":
          out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D) # [1, 2, 16, 128, 128]
          out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          return out_scale_1_8__3D

        elif self.out_scale=="1_4":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D)
          out = torch.cat((out, _skip_1_4), 1)
          out = F.relu(self.conv1_4(out))
          out_scale_1_4__2D = self.conv_out_scale_1_4(out)

          out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D) # [1, 20, 16, 128, 128]
          out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          return out_scale_1_4__3D

        elif self.out_scale=="1_2":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D) # [1, 4, 64, 64] 1/4
          # if self.uncertain:
          out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D) # [1, 3, 4, 32, 32]
          out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 4, 2)
          #   out_scale_1_8_4__3D = self.upconv4(out_scale_1_8__3D)
          out = torch.cat((out, _skip_1_4), 1) # [1, 68, 64, 64] 
          out = F.relu(self.conv1_4(out)) # [1, 64, 64, 64] 
          out_scale_1_4__2D = self.conv_out_scale_1_4(out) # [1, 8, 64, 64] 
          # if self.uncertain:
          out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D)  # [1, 3, 8, 64, 64]
          out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 4, 2)
            # out_scale_1_4__3D = self.skip_conv4(torch.cat((out_scale_1_4__3D, out_scale_1_8_4__3D), axis=1))
            # out_scale_1_4_2__3D = self.upconv3(out_scale_1_4__3D)
          # Out 1_2
          out = self.deconv1_4(out_scale_1_4__2D) # [1, 8, 128, 128] 
          out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1) # [1, 60, 128, 128]
          out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
          out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])
          # if self.uncertain:
            
          out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D) # [1, 3, 16, 128, 128]
            # out_scale_1_2__3D = self.skip_conv3(torch.cat((out_scale_1_2__3D, out_scale_1_4_2__3D), axis=1))
            
          out_scale_1_2__3D = out_scale_1_2__3D.permute(0, 1, 3, 4, 2) # [1, 2, 128, 128, 16]
          # out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 4, 2)
          # out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 4, 2)
          # if self.uncertain:
          return out_scale_1_2__3D, out_scale_1_4__3D, out_scale_1_8__3D
          # else:
          #   return out_scale_1_2__3D

        elif self.out_scale=="1_1":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D)
          print('out.shape', out.shape)  # [1, 4, 64, 64]
          out = torch.cat((out, _skip_1_4), 1)
          out = F.relu(self.conv1_4(out))
          out_scale_1_4__2D = self.conv_out_scale_1_4(out)
          # print('out_scale_1_4__2D.shape', out_scale_1_4__2D.shape)  # [1, 8, 64, 64]

          # Out 1_2
          out = self.deconv1_4(out_scale_1_4__2D)
          print('out.shape', out.shape)  # [1, 8, 128, 128]
          out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
          out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
          out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])
          # print('out_scale_1_2__2D.shape', out_scale_1_2__2D.shape)  # [1, 16, 128, 128]

          # Out 1_1
          out = self.deconv1_2(out_scale_1_2__2D)
          out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
          out_scale_1_1__2D = F.relu(self.conv1_1(out)) # [bs, 32, 256, 256]

          out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)
          # Take back to [W, H, D] axis order
          out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 4, 2)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
          return out_scale_1_1__3D


    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))

        #compressing bit flags.
        # yapf: disable
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        # yapf: enable

        return np.array(compressed, dtype=np.uint8)

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.foward_training(**kwargs)
        else:
            return self.foward_test(**kwargs)


    # @auto_fp16(apply_to=('img', 'points'))
    def foward_training(self,
                        img_metas=None,
                        img=None,
                        target=None):


        # for binary classification
        # if self.uncertain:
        target_1 = target[0]
        target_2 = target[1]
        target_3 = target[2]
        ones_1 = torch.ones_like(target_1).to(target_1.device)
        ones_2 = torch.ones_like(target_2).to(target_2.device)
        ones_3 = torch.ones_like(target_3).to(target_3.device)
        target_1 = torch.where(torch.logical_or(target_1==255, target_1==0), target_1, ones_1)
        target_2 = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones_2)
        target_3 = torch.where(torch.logical_or(target_3==255, target_3==0), target_3, ones_3)
        # else:
        #   target = target[0]
        #   ones = torch.ones_like(target).to(target.device)
        #   target = torch.where(torch.logical_or(target==255, target==0), target, ones) # [1, 128, 128, 16]
        # target[target==255] = 2

        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas] # [dict(), dict(), ...] 

        if self.dataset == 'KITTI':
          depth =  torch.from_numpy(img_metas[0]["pseudo_pc"]).reshape(256, 256, 32).unsqueeze(0) # [1, 256, 256, 32] 
        else:
          depth =  torch.from_numpy(img_metas[0]["pseudo_pc"]).reshape(200, 200, 16).unsqueeze(0) # [1, 256, 256, 32]
        
        # if self.uncertain:
        out_level_1,  out_level_2, out_level_3 = self.step(depth.permute(0, 3, 1, 2).to(target_1.device))
        if self.uncertain:
          pred_sigma_1 = out_level_1[:,2,:,:,:]
          pred_sigma_2 = out_level_2[:,2,:,:,:]
          pred_sigma_3 = out_level_3[:,2,:,:,:]
        else:
          pred_sigma_1 = pred_sigma_2 = pred_sigma_3 = None
        out_1 = out_level_1[:,0:2,:,:,:]
        out_2 = out_level_2[:,0:2,:,:,:]
        out_3 = out_level_3[:,0:2,:,:,:]  
        # calculate loss
        losses = dict()
        losses_pts = dict()
        class_weights_level_1 = self.class_weights_level_1.type_as(target_1)
        class_weights_level_2 = self.class_weights_level_1.type_as(target_2)
        class_weights_level_3 = self.class_weights_level_1.type_as(target_3)
        loss_sc_level_1 = BCE_ssc_loss(out_1, target_1, class_weights_level_1, self.alpha)
        loss_sc_level_2 = BCE_ssc_loss(out_2, target_2, class_weights_level_2, self.alpha)
        loss_sc_level_3 = BCE_ssc_loss(out_3, target_3, class_weights_level_3, self.alpha)
        losses_pts['loss_sc_level_1'] = 1 * loss_sc_level_1
        losses_pts['loss_sc_level_2'] = 0.9 * loss_sc_level_2
        losses_pts['loss_sc_level_3'] = 0.81 * loss_sc_level_3
        losses.update(losses_pts)
        # print((torch.argmax(out_1[0], 0) == 1).nonzero().shape)
        # print((torch.argmax(out_2[0], 0) == 1).nonzero().shape)
        # print((torch.argmax(out_3[0], 0) == 1).nonzero().shape)
          
        # else:
        #   out_level_1 = self.step(depth.permute(0, 3, 1, 2).to(target.device))
        #   # calculate loss
        #   losses = dict()
        #   losses_pts = dict()
        #   class_weights_level_1 = self.class_weights_level_1.type_as(target)
        #   loss_sc_level_1 = BCE_ssc_loss(out_level_1, target, class_weights_level_1, self.alpha)
        #   losses_pts['loss_sc_level_1'] = loss_sc_level_1          
        #   losses.update(losses_pts)

        return losses

    def foward_test(self,
                        img_metas=None,
                        sequence_id=None,
                        img=None,
                        target=None,
                        T_velo_2_cam=None,
                        cam_k=None, **kwargs):

        # 07/12/2022, Yiming Li, only support batch_size = 1

        # for binary classification
        target = target[1]
        ones = torch.ones_like(target).to(target.device)
        target = torch.where(torch.logical_or(target==255, target==0), target, ones) # [1, 128, 128, 16]

        # target[target==255] = 2

        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas] # [dict(), dict(), ...] 
        if self.dataset == 'KITTI':
          depth =  torch.from_numpy(img_metas[0]["pseudo_pc"]).reshape(256, 256, 32).unsqueeze(0) # [1, 256, 256, 32] 
        else:
          depth =  torch.from_numpy(img_metas[0]["pseudo_pc"]).reshape(200, 200, 16).unsqueeze(0) # [1, 256, 256, 32]
        if self.uncertain:
          ssc_pred_1, ssc_pred_2, ssc_pred_3 = self.step(depth.permute(0, 3, 1, 2).to(target.device))
        else:
          ssc_pred_1, ssc_pred_2, ssc_pred_3 = self.step(depth.permute(0, 3, 1, 2).to(target.device))

        y_pred_1 = ssc_pred_1.detach().cpu().numpy() # [1, 20, 128, 128, 16]
        y_pred_2 = ssc_pred_2.detach().cpu().numpy()
        y_pred_3 = ssc_pred_3.detach().cpu().numpy()
        if self.uncertain:
          pred_sigma = y_pred[:,2,:,:,:]
          y_pred = y_pred[:,0:2,:,:,:]
          print(pred_sigma.nonzero()[0].shape[0])
        y_pred_1 = np.argmax(y_pred_1, axis=1).astype(np.uint8) # [1, 128, 128, 16]
        y_pred_2 = np.argmax(y_pred_2, axis=1).astype(np.uint8) # [1, 128, 128, 16]
        y_pred_3 = np.argmax(y_pred_3, axis=1).astype(np.uint8) # [1, 128, 128, 16]

        #save query proposal 
        img_path = img_metas[0]['img_filename'] 
        frame_id = os.path.splitext(img_path[0])[0][-6:]

        # msnet3d
        # query_2_root = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries_2')
        # if not os.path.exists(query_2_root):
        #     os.makedirs(query_2_root)

        query_4_root = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries_4')
        if not os.path.exists(query_4_root):
            os.makedirs(query_4_root)

        query_8_root = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries_8')
        if not os.path.exists(query_8_root):
            os.makedirs(query_8_root)
       
        # save_query_path = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries', frame_id + ".query_iou5203_pre7712_rec6153")
        # save !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if self.dataset == "nuscenes":
        #   token = img_metas[0]["img_filename"][0].split("/")[-1].replace(".png", ".npy")
        #   if self.uncertain:
        #     save_query_path = os.path.join("./nuscenes/trainval/dataset/midas-uncertain", img_metas[0]['sequence_id'], token)
        #     pred_sigma_path = save_query_path.replace(".npy", "_sigma.npy")
        #   else:
        #     save_query_path = os.path.join("./nuscenes/trainval/dataset/midas-sweep0", img_metas[0]['sequence_id'], token)
        #     save_query_path_1 = save_query_path.replace(".npy", "_1.npy")
        #     save_query_path_2 = save_query_path.replace(".npy", "_2.npy")
        #     save_query_path_3 = save_query_path.replace(".npy", "_3.npy")
        #   save_dir = os.path.dirname(save_query_path)
        #   print(save_dir)
        #   os.makedirs(save_dir, exist_ok=True)
          # np.save(save_query_path_1, y_pred_1)
          # np.save(save_query_path_2, y_pred_2)
          # np.save(save_query_path_3, y_pred_3)
          # print(y_pred_1.shape, y_pred_2.shape, y_pred_3.shape)
          # if self.uncertain:
          #   np.save(pred_sigma_path, pred_sigma)
          
        # else:
        save_query_path_4 = os.path.join(query_4_root, frame_id + ".query_iou5203_pre7712_rec6153")
        y_pred_bin_4 = self.pack(y_pred_2)
        print(save_query_path_4, y_pred_bin_4.shape)
        # y_pred_bin_4.tofile(save_query_path_4)
        
        save_query_path_8 = os.path.join(query_8_root, frame_id + ".query_iou5203_pre7712_rec6153")
        y_pred_bin_8 = self.pack(y_pred_3)
        print(save_query_path_8, y_pred_bin_8.shape)
        # y_pred_bin_8.tofile(save_query_path_8)
        #-------------------------------------------------------------------------------------------------
        # print(img_metas[0]['sequence_id'])
        # print(img_metas[0]["img_filename"])
        result = dict()
        y_true = target.cpu().numpy()
        result['y_pred'] = y_pred_2
        result['y_true'] = y_true
        return result

class UpConv3D(nn.Module):
    """
    Use bilinear followed by conv
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(UpConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        x = self.non_linear(x)
        return x
    
class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SkipConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.non_linear(x)
        return x

class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list, uncertain=False):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    # if uncertain:
    #   self.conv_classes = nn.Conv3d(planes, nbr_classes + 1, kernel_size=3, padding=1, stride=1)
    # else:
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)
    x_in = self.relu(x_in)

    return x_in
