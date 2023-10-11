# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Header(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        self.open_mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, 768),
        )
        self.lidar_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_3d =  nn.Sequential(nn.Conv3d(in_channels=feature, out_channels=feature, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(feature), nn.ReLU())
    
        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(128), nn.ReLU())

    def forward(self, input_dict, img_metas, openset=None, ms=None, lidar=None):
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        # x3d_l1: 1/2 , x3d_up_l1: full, x3d_down_l1: 
        x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        # debug without conv
        # x3d_up_l1 = self.conv_3d(x3d_up_l1)
        # x3d_up_save =x3d_up_l1.detach().cpu()
        if ms:
            x3d_down_l1 = self.downsample(x3d_l1)
            x3d_down_l2 = self.downsample(x3d_down_l1)
        
        _, feat_dim, w, l, h  = x3d_up_l1.shape
        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        if openset: 
            openssc_logit_full = self.open_mlp_head(x3d_up_l1)

        ssc_logit_full = self.mlp_head(x3d_up_l1)
        if ms:
            x3d_l1 = x3d_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
            x3d_down_l1 = x3d_down_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
            x3d_down_l2 = x3d_down_l2.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        
            ssc_logit_2 = self.mlp_head(x3d_l1)
            ssc_logit_4 = self.mlp_head(x3d_down_l1)
            ssc_logit_8 = self.mlp_head(x3d_down_l2)
        
        # _, feat_dim, w, l, h  = x3d_l1.shape
        # x3d_l1 = x3d_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        # ssc_logit_full = self.mlp_head(x3d_l1)
        
        # TODO check why lidar not work
        if lidar != None:
            lidar = lidar.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # TODO get logits for lidar
            lidar_out = F.grid_sample(x3d_up_l1.reshape(w,l,h,feat_dim).permute(3,0,1,2).unsqueeze(0), lidar, mode='bilinear')
            lidar_out = lidar_out.squeeze(2).squeeze(2).squeeze(0).permute(1,0) 
            lidar_out = self.lidar_head(lidar_out)
            n, _ = lidar_out.shape  
            res["lidar_out"] = lidar_out.reshape(n, self.class_num).permute(1,0).unsqueeze(0)
        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
        res["bev"] = input_dict["x3d"]
        # res["bev_full"] = x3d_up_save
        if ms:
            res["ssc_logit_2"] = ssc_logit_2.reshape(w // 2, l // 2, h // 2, self.class_num).permute(3,0,1,2).unsqueeze(0)
            res["ssc_logit_4"] = ssc_logit_4.reshape(w // 4 , l // 4, h // 4, self.class_num).permute(3,0,1,2).unsqueeze(0)
            res["ssc_logit_8"] = ssc_logit_8.reshape(w // 8, l // 8, h // 8, self.class_num).permute(3,0,1,2).unsqueeze(0)
        if openset:
            res['openssc_logit'] = openssc_logit_full.reshape(w, l, h, 768).permute(3,0,1,2).unsqueeze(0)
        
        return res

class PixelShuffle3D(nn.Module):
    """
    3D pixelShuffle
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: int
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)