# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MonoOcc/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.MonoOcc.utils.header import Header
from projects.mmdet3d_plugin.MonoOcc.utils.header_ms import Header_ms
from projects.mmdet3d_plugin.MonoOcc.utils.occupied_header import Occupied_Header
from projects.mmdet3d_plugin.MonoOcc.utils.ssc_loss import sem_scal_loss, KL_sep, geo_scal_loss, CE_ssc_loss, CE_loss_2D, BCE_ssc_loss, CE_lidar_loss
from projects.mmdet3d_plugin.MonoOcc.modules.cvt import CVT
from projects.mmdet3d_plugin.MonoOcc.modules.sem import Sem_Decoder
# from projects.mmdet3d_plugin.models.utils.bricks import run_time

@HEADS.register_module()
class MonoOccHead_ms(nn.Module):
    def __init__(
        self,
        *args,
        dataset,
        bev_h,
        bev_w,
        bev_z,
        real_w,
        real_h,
        real_z,
        voxel_origin,
        classes,
        classes_names,
        classes_weights,
        cross_transformer,
        self_transformer,
        positional_encoding,
        embed_dims,
        ratio_2D,
        use_cvt=False,
        use_sem=False,
        look_ahead=False,
        fine_grain=False,
        use_lidar=False,
        ms=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        # for multi-task 1/2 resolution
        self.bev_h = []
        self.bev_w = []
        self.bev_z = []
        self.real_h = real_h
        self.real_w = real_w
        self.real_z = real_z
        self.bev_embed = []
        self.mask_embed = []
        self.positional_encoding = []
        self.H = []
        self.W = []
        self.cross_transformer = []
        self.self_transformer = []
        self.voxel_origin = voxel_origin
        self.n_classes = classes
        self.embed_dims = embed_dims
        self.ratio_2D = ratio_2D
        for i in range(3):
            self.bev_h.append(bev_h // (2 ** i))
            self.bev_w.append(bev_w // (2 ** i))
            self.bev_z.append(bev_z // (2 ** i))
            self.bev_embed.append(nn.Embedding(bev_h // (2 ** i) * bev_w // (2 ** i) * bev_z // (2 ** i), self.embed_dims))
            self.mask_embed.append(nn.Embedding(1, self.embed_dims))
            self.positional_encoding.append(build_positional_encoding(positional_encoding))
            self.H.append(positional_encoding["row_num_embed"])
            self.W.append(positional_encoding["col_num_embed"])
            if i == 0:
                positional_encoding["row_num_embed"] = positional_encoding["row_num_embed"] // 2
                positional_encoding["col_num_embed"] = positional_encoding["col_num_embed"] // 4
            elif i == 1:
                positional_encoding["row_num_embed"] = positional_encoding["row_num_embed"] // 4
                positional_encoding["col_num_embed"] = positional_encoding["col_num_embed"] // 2
            self.cross_transformer.append(build_transformer(cross_transformer))
            self.self_transformer.append(build_transformer(self_transformer))
    

        
        # add cross transformer
        self.cross_transformer_late = build_transformer(cross_transformer)
        # add cross view transformer
        if use_cvt:
            self.cvt = CVT(input_channel=self.embed_dims, downsample_ratio=2, iter_num=3)
        if use_sem:
            self.sem_decoder = Sem_Decoder(input_channel=self.embed_dims, num_classes=self.n_classes, ratio=2)     
        self.header = Header_ms(self.n_classes, nn.BatchNorm3d, feature=self.embed_dims)
        if fine_grain:
            self.occupied_header = Occupied_Header(2, nn.BatchNorm3d, feature=self.embed_dims)
            # self.occ_embed = nn.Embedding(2 * self.bev_h * 2 * self.bev_w * 2 * self.bev_z, self.embed_dims)
            
        self.classes_names = classes_names
        self.class_weights = torch.from_numpy(np.array(classes_weights))
        self.bce_class_weights = torch.from_numpy(np.array((0.42, 0.58)))
        self.use_cvt = use_cvt
        self.use_sem = use_sem
        self.look_ahead = look_ahead
        self.fine_grain = fine_grain
        self.use_lidar = use_lidar
        self.ms = ms
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        
    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        bs, num_cam, C, H, W = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
         # Cross-View Transformer for img features
        if self.use_cvt:
            for i in range(len(mlvl_feats)):
                # print(mlvl_feats[0].shape)
                mlvl_feats[i] = self.cvt(mlvl_feats[i])
        
        if self.use_sem:
            sem_out_list = []
            for i in range(len(mlvl_feats)):
                sem_out = self.sem_decoder(mlvl_feats[i])
                sem_out_list.append(sem_out)
                
        vox_feats_diffs = []
        for i in range(2, -1, -1):
            bev_queries = self.bev_embed[i].weight.to(dtype).to(mlvl_feats[0].device) #[128*128*16, dim]
            # Generate bev postional embeddings for cross and self attention
            self.positional_encoding[i].to(mlvl_feats[0].device)
            self.cross_transformer[i].to(mlvl_feats[0].device)
            self.self_transformer[i].to(mlvl_feats[0].device)
            bev_pos_cross_attn = self.positional_encoding[i](torch.zeros((bs, self.H[i], self.W[i]), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
            bev_pos_self_attn = self.positional_encoding[i](torch.zeros((bs, self.H[i], self.W[i]), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

    
            # Load query proposals
            proposal =  img_metas[0]['proposal'][i].reshape(self.bev_h[i], self.bev_w[i], self.bev_z[i])
            unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
            masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
               
            vox_coords, ref_3d = self.get_ref_3d(self.bev_h[i], self.bev_w[i], self.bev_z[i])

            # Compute seed features of query proposals by deformable cross attention
            seed_feats = self.cross_transformer[i].get_vox_features(
                mlvl_feats, 
                bev_queries,
                self.bev_h[i],
                self.bev_w[i],
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=(self.real_h / self.bev_h[i], self.real_w / self.bev_w[i]),
                bev_pos=bev_pos_cross_attn,
                img_metas=img_metas,
                prev_bev=None,
            )

            # Complete voxel features by adding mask tokens
            vox_feats = torch.empty((self.bev_h[i], self.bev_w[i], self.bev_z[i], self.embed_dims), device=bev_queries.device)
            vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
            vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
            vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed[i].weight.to(mlvl_feats[0].device).view(1, self.embed_dims).expand(masked_idx.shape[1], self.embed_dims).to(dtype)

            # Diffuse voxel features by deformable self attention
            vox_feats_diff = self.self_transformer[i].diffuse_vox_features(
                mlvl_feats,
                vox_feats_flatten,
                self.H[i],
                self.W[i],
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=(self.real_h / self.bev_h[i], self.real_w / self.bev_w[i]),
                bev_pos=bev_pos_self_attn,
                img_metas=img_metas,
                prev_bev=None,
            )

            if self.look_ahead:
                late_unmasked_idx = np.array([i for i in range(self.bev_h[i] * self.bev_w[i] * self.bev_z[i])], dtype=np.int32)
                late_unmasked_idx = np.expand_dims(late_unmasked_idx, 0)
                    
                voxel_feats_late = self.cross_transformer_late.get_vox_features(
                    mlvl_feats, 
                    vox_feats_diff[0],
                    self.bev_h[i],
                    self.bev_w[i],
                    ref_3d=ref_3d,
                    vox_coords=vox_coords,
                    unmasked_idx=late_unmasked_idx,
                    grid_length=(self.real_h / self.bev_h[i], self.real_w / self.bev_w[i]),
                    bev_pos=bev_pos_cross_attn,
                    img_metas=img_metas,
                    prev_bev=None,            
                )

                vox_feats_diff = voxel_feats_late.reshape(self.bev_h[i], self.bev_w[i], self.bev_z[i], self.embed_dims)
            else:
                vox_feats_diff = vox_feats_diff.reshape(self.bev_h[i], self.bev_w[i], self.bev_z[i], self.embed_dims)
        
            vox_feats_diffs.append(vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0))
        
        input_dict = {
            "x3d": vox_feats_diffs,
        }

        lidar = None
        if self.use_lidar:
            lidar = torch.tensor(img_metas[0]["lidar"], device=bev_queries.device)
            lidar[:, 0] = lidar[:, 0] / (self.real_w3) * 2 - 1
            lidar[:, 1] = (lidar[:, 1] / (self.real_h3)) * 2
            lidar[:, 2] = ((lidar[:, 2] + 2.) / (self.real_z3)) * 2 - 1
        out = self.header(input_dict, img_metas, lidar)
        if self.use_sem:
            out["semantic"] = sem_out_list
            
        return out 

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """
        # if self.fine_grain:
        #     occ_pred = out_dict["occupied"]
        ssc_pred = out_dict["ssc_logit"]
        if self.ms:
            ssc_pred_2 = out_dict["ssc_logit_2"]
            ssc_pred_4 = out_dict["ssc_logit_4"]
            ssc_pred_8 = out_dict["ssc_logit_8"]
            ssc_preds = [ssc_pred, ssc_pred_2, ssc_pred_4, ssc_pred_8]
            target_2 = torch.tensor(img_metas[0]["target_2"], device=ssc_pred.device).unsqueeze(0)
            target_4 = torch.tensor(img_metas[0]["target_4"], device=ssc_pred.device).unsqueeze(0)
            target_8 = torch.tensor(img_metas[0]["target_8"], device=ssc_pred.device).unsqueeze(0)
            targets = [target, target_2, target_4, target_8]
        else:
            ssc_preds = [ssc_pred]
            targets = [target]
        
        if self.use_sem:
            sem_pred = out_dict["semantic"]
            target_2d = torch.tensor(img_metas[0]["semantic"], device=ssc_pred.device) 
            
        if self.use_lidar:
            lidar_out = out_dict["lidar_out"]
            target_lidar = torch.tensor(img_metas[0]["lidar_label"], device=ssc_pred.device).permute(1,0)
        
        if step_type== "train":
            loss_dict = dict()
            bce_class_weight = self.bce_class_weights.type_as(target)
            class_weight = self.class_weights.type_as(target)
            if self.use_sem:
                ce_loss = CE_loss_2D(sem_pred, target_2d, self.ratio_2D)
                loss_dict['loss_2d_ce'] = ce_loss
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_preds, targets, class_weight)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_preds, targets)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_preds, targets)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            # if self.fine_grain:
            #     loss_bce = BCE_ssc_loss(occ_pred, target_occ, bce_class_weight)
            #     loss_dict['loss_bce'] = loss_bce
            
            if self.use_lidar:
                loss_lidar = CE_lidar_loss(lidar_out, target_lidar, class_weight)
                loss_dict['lidar'] = loss_lidar

            return loss_dict

        elif step_type== "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            if self.save_flag:
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")

    def get_ref_3d(self, bev_h=None, bev_w=None, bev_z=None):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (self.real_h, self.real_w, self.real_z)
        vox_origin = np.array(self.voxel_origin)
        voxel_size = self.real_h / bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        if self.dataset == "KITTI":
            vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T
        elif self.dataset == "nuScenes":
            vox_coords = np.concatenate([yv.reshape(1,-1), xv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T
        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/bev_h, (yv.reshape(1,-1)+0.5)/bev_w, (zv.reshape(1,-1)+0.5)/bev_z,], axis=0).astype(np.float64).T 

        return vox_coords, ref_3d

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./MonoOcc", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
