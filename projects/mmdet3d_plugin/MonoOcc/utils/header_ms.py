# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Header_ms(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(Header_ms, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        self.lidar_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_3d_4 =  nn.Sequential(nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(128), nn.ReLU())
        self.conv_3d_2 =  nn.Sequential(nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(128), nn.ReLU())
        self.conv_3d_1 =  nn.Sequential(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(128), nn.ReLU())  
        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(128), nn.ReLU())

    def forward(self, input_dict, img_metas, lidar=None):
        res = {}

        x3d_l8, x3d_l4, x3d_l2 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        # x3d_l1: 1/2 , x3d_up_l1: full, x3d_down_l1: 
        x3d_up_l4 = self.up_scale_2(x3d_l8) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        x3d_l4 = self.conv_3d_4(torch.cat((x3d_up_l4, x3d_l4), dim=1))
        
        x3d_up_l2 = self.up_scale_2(x3d_l4)
        x3d_l2 = self.conv_3d_2(torch.cat((x3d_up_l2, x3d_l2), dim=1))
        
        x3d_up_l1 = self.up_scale_2(x3d_l2)
        x3d_l1 = self.conv_3d_1(x3d_up_l1)

        
        _, feat_dim, w, l, h  = x3d_l1.shape
        x3d_l1 = x3d_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        x3d_l2 = x3d_l2.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        x3d_l4 = x3d_l4.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        x3d_l8 = x3d_l8.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        
        ssc_logit_full = self.mlp_head(x3d_l1)
        ssc_logit_2 = self.mlp_head(x3d_l2)
        ssc_logit_4 = self.mlp_head(x3d_l4)
        ssc_logit_8 = self.mlp_head(x3d_l8)
        
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
        res["ssc_logit_2"] = ssc_logit_2.reshape(w // 2, l // 2, h // 2, self.class_num).permute(3,0,1,2).unsqueeze(0)
        res["ssc_logit_4"] = ssc_logit_4.reshape(w // 4 , l // 4, h // 4, self.class_num).permute(3,0,1,2).unsqueeze(0)
        res["ssc_logit_8"] = ssc_logit_8.reshape(w // 8, l // 8, h // 8, self.class_num).permute(3,0,1,2).unsqueeze(0)
        
        return res