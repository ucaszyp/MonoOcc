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
from projects.mmdet3d_plugin.MonoOcc.utils.occupied_header import Occupied_Header
from projects.mmdet3d_plugin.MonoOcc.utils.ssc_loss import sem_scal_loss, KL_sep, geo_scal_loss, CE_ssc_loss, CE_loss_2D, BCE_ssc_loss, CE_lidar_loss, cos_similarity, Distill_loss, silog_loss, cos_similarity_2d
from projects.mmdet3d_plugin.MonoOcc.modules.cvt import CVT
from projects.mmdet3d_plugin.MonoOcc.modules.tvt import TVT
from projects.mmdet3d_plugin.MonoOcc.modules.sem import Sem_Decoder
from projects.mmdet3d_plugin.MonoOcc.modules.depth import Depth_Decoder
from projects.mmdet3d_plugin.MonoOcc.modules.large_model import Large_Decoder
# from projects.mmdet3d_plugin.MonoOcc.utils.clip_embedding import obtain_text_features_and_palette
from projects.mmdet3d_plugin.MonoOcc.utils.utils import feature_add_position
@HEADS.register_module()
class MonoOccHead(nn.Module):
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
        ratio_2D, # 2d semantic, cross entropy
        ratio_3D, # 3d ssc, cross entropy
        ratio_sem, # sem_scal loss
        ratio_geo, # geo scal loss
        ratio_distill, # distill loss
        ratio_2d_distill, # large model distill loss
        use_cvt=False,
        use_sem=False,
        look_ahead=False,
        fine_grain=False,
        use_lidar=False,
        use_depth=False,
        use_large_model=False,
        ms=False,
        distill=False,
        openset=False,
        extractor="openseg",
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_h = real_h
        self.real_w = real_w
        self.real_z = real_z
        self.voxel_origin = voxel_origin
        self.n_classes = classes
        self.embed_dims = embed_dims
        self.ratio_2D = ratio_2D
        self.ratio_3D = ratio_3D
        self.ratio_sem = ratio_sem
        self.ratio_geo = ratio_geo
        self.ratio_distill = ratio_distill
        self.ratio_2d_distill = ratio_2d_distill
        self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        # self.occ_encoding = build_positional_encoding(occ_encoding)
        self.H = positional_encoding["row_num_embed"]
        self.W = positional_encoding["col_num_embed"]
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        # add cross transformer
        self.cross_transformer_late = build_transformer(cross_transformer)
        # add cross view transformer
        if use_cvt:
            # self.cvt = CVT(input_channel=self.embed_dims, downsample_ratio=2, iter_num=3)
            self.cvt = TVT(num_layers=6,
                           d_model=128,
                           nhead=1,
                           ffn_dim_expansion=4)
        if use_sem:
            self.sem_decoder = Sem_Decoder(input_channel=self.embed_dims, num_classes=self.n_classes, ratio=2)     
        if use_depth:
            self.depth_decoder = Depth_Decoder(input_channel=self.embed_dims, num_classes=1, ratio=2)
        # NOTE:2023/9/4
        if use_large_model:
            self.open_fc = nn.Sequential(
                nn.Linear(self.embed_dims, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
            )
            self.large_decoder = Large_Decoder(input_channel=self.embed_dims, ratio=2)
        
        self.header = Header(self.n_classes, nn.BatchNorm3d, feature=self.embed_dims)
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
        self.use_depth = use_depth
        self.use_large_model = use_large_model
        self.ms = ms
        self.distill = distill
        self.openset = openset
        self.extractor = extractor
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        # for openset
        if self.openset:
            # self.text_features = obtain_text_features_and_palette(dataset, extractor).type(torch.float32)
            pass
        self.mlp = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, self.embed_dims),
        )
            
        
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
        bev_queries = self.bev_embed.weight.to(dtype) #[128*128*16, dim]
        if self.openset:
            mlvl_feats[0] = mlvl_feats[0].permute(0,1,3,4,2).reshape(-1, C)
            mlvl_feats[0] = self.mlp(mlvl_feats[0])
            mlvl_feats[0] = mlvl_feats[0].reshape(bs, num_cam, H, W, self.embed_dims).permute(0,1,4,2,3)
        # Cross-View Transformer for img features
        if self.use_cvt:
              for i in range(len(mlvl_feats)):
                  b, n, c, h, w = mlvl_feats[i].shape
                  if h % 2 == 0 and w % 2 == 0:
                  # print(c,h,w)
                      for j in range(n-1):
                          feat0 = mlvl_feats[i][:,j,:,:,:]
                          feat1 = mlvl_feats[i][:,j+1,:,:,:]
                          feat0, feat1 = feature_add_position(feat0, feat1, 2, 128)
                          mlvl_feats[i][:,j,:,:,:], mlvl_feats[i][:,j+1,:,:,:] = self.cvt(feat0, feat1, attn_type="swin", attn_num_splits=2)

        
        if self.use_sem:
            sem_out_list = []
            for i in range(len(mlvl_feats)):
                sem_out = self.sem_decoder(mlvl_feats[i])
                sem_out_list.append(sem_out)
        if self.use_depth:
            depth_out_list = []
            for i in range(len(mlvl_feats)):
                depth_out = self.depth_decoder(mlvl_feats[i])
                depth_out_list.append(depth_out)
        # NOTE:2023/9/4
        if self.use_large_model:
            # img_feat = self.large_decoder(mlvl_feats[0])
            # _, _, _, h, w = img_feat.shape
            # img_feat = img_feat.permute(0,1,3,4,2).reshape(-1, C)
            img_feat = mlvl_feats[0].permute(0,1,3,4,2).reshape(-1, C)
            img_feat = self.open_fc(img_feat)
            img_feat = img_feat.reshape(bs, num_cam, H, W, 512).permute(0,1,4,2,3)
        if self.openset:
            self.text_features = self.text_features.to(mlvl_feats[0].device)
                # semantic = img_metas[0]["semantic"]
        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, self.H, self.W), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, self.H, self.W), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        # if self.fine_grain:
        #     bev_pos_occ_attn = self.occ_encoding(torch.zeros((bs, 2 * self.H, 4 * self.W), device=bev_queries.device).to(dtype)).to(dtype) 
        # Load query proposals
        proposal =  img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        
        # unmasked_len =  unmasked_idx.shape[1]
        # unmasked_selected = np.expand_dims(np.random.choice([False, True], unmasked_len, p=[0.2, 0.8]), 0)
        
        # masked_len =  masked_idx.shape[1]
        # masked_selected = np.expand_dims(np.random.choice([False, True], masked_len, p=[0.9, 0.1]), 0)  
          
        # unmasked_idx_new = np.concatenate((unmasked_idx[unmasked_selected], masked_idx[masked_selected]), -1)
        # unmasked_idx_new = np.expand_dims(np.sort(unmasked_idx_new), 0)
        
        # masked_idx_new = np.concatenate((unmasked_idx[~unmasked_selected], masked_idx[~masked_selected]), -1)
        # masked_idx_new = np.expand_dims(np.sort(masked_idx_new), 0)
        
        # unmasked_idx = unmasked_idx_new
        # masked_idx = masked_idx_new
        # unmasked_idx[unmasked_selected] 
        
        vox_coords, ref_3d = self.get_ref_3d(self.bev_h, self.bev_w, self.bev_z)

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats, 
            bev_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        # print("cross attention 1 nan: ", torch.isnan(seed_feats).sum())

        # Complete voxel features by adding mask tokens
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[1], self.embed_dims).to(dtype)

        # Diffuse voxel features by deformable self attention
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            self.H,
            self.W,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        # print("self attention 1 nan: ", torch.isnan(vox_feats_diff).sum())
        # out_occ = None
        if self.look_ahead:
            late_unmasked_idx = np.array([i for i in range(self.bev_h * self.bev_w * self.bev_z)], dtype=np.int32)
            late_unmasked_idx = np.expand_dims(late_unmasked_idx, 0)
            # if self.fine_grain:     
            #     out_occ = self.occupied_header(vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims))
            #     occupied_mask = torch.argmax(out_occ, dim=1)
            #     occupied_mask = occupied_mask.reshape(self.bev_h, self.bev_w, self.bev_z)
            #     occupied_mask = occupied_mask.detach().cpu().numpy()
            #     late_unmasked_idx = np.asarray(np.where(occupied_mask.reshape(-1) > 0)).astype(np.int32)
            #     # late_unmasked_idx = occupied_mask
            #     late_masked_idx = np.asarray(np.where(occupied_mask.reshape(-1) == 0)).astype(np.int32)
                
            voxel_feats_late = self.cross_transformer_late.get_vox_features(
                mlvl_feats, 
                vox_feats_diff[0],
                self.bev_h,
                self.bev_w,
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=late_unmasked_idx,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos_cross_attn,
                img_metas=img_metas,
                prev_bev=None,            
            )
            # if self.fine_grain:
            #     # vox_feats_diff[:, vox_coords[late_unmasked_idx[0], 3], :] = voxel_feats_late
            #     # vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
            #     vox_fine_flatten = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
            #     vox_fine_flatten = vox_fine_flatten.reshape(-1, self.embed_dims)
            #     vox_fine_flatten[vox_coords[late_unmasked_idx[0], 3], :] = voxel_feats_late[0]
            #     vox_fine_flatten[vox_coords[late_masked_idx[0], 3], :] = vox_feats_diff[0][vox_coords[late_masked_idx[0], 3], :]
            #     vox_feats_diff = vox_fine_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
            # else:
            # print("cross attention 2 nan: ", torch.isnan(voxel_feats_late).sum())
            vox_feats_diff = voxel_feats_late.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
        else:
            vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
            
        input_dict = {
            "x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0),
        }
        # if self.use_implicit:
        #     out = self.occ_header(input_dict)
            
        # else:
        #     out = self.header(input_dict)
        # if self.fine_grain:
        #     out, vox_feats_diff = self.occupied_header(input_dict)
        #     # TODO: generate mask from out
        #     occupied_mask = torch.argmax(out, dim=1)
        #     occupied_mask = occupied_mask.reshape(2 * self.bev_h, 2 * self.bev_w, 2 * self.bev_z)
        #     occupied_mask = occupied_mask.detach().cpu().numpy()
        #     occupied_mask = np.asarray(np.where(occupied_mask.reshape(-1) > 0)).astype(np.int32)
        #     # TODO: generate bev postional embeddings for new cross attn, done
        #     # TODO: cross attn
        #     vox_coords_occ, ref_3d_occ = self.get_ref_3d(self.bev_h * 2, self.bev_w * 2, self.bev_z * 2)
        #     voxel_feats_fine = self.cross_transformer_late.get_vox_features(
        #         mlvl_feats, 
        #         vox_feats_diff,
        #         2 * self.bev_h,
        #         2 * self.bev_w,
        #         ref_3d=ref_3d_occ,
        #         vox_coords=vox_coords_occ,
        #         unmasked_idx=occupied_mask,
        #         grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        #         bev_pos=bev_pos_occ_attn,
        #         img_metas=img_metas,
        #         prev_bev=None,            
        #     )
        #     vox_feats_diff = voxel_feats_fine.reshape(self.bev_h * 2, self.bev_w * 2, self.bev_z * 2, self.embed_dims)
        # input_dict = {
        #     "x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0),
        # }
        # save for test
        # torch.save(input_dict['x3d'][0].detach().cpu(), save_path)
        # print(save_path)
        lidar = None
        if self.use_lidar:
            lidar = torch.tensor(img_metas[0]["lidar"], device=bev_queries.device)
            lidar[:, 0] = lidar[:, 0] / (self.real_w) * 2 - 1
            lidar[:, 1] = (lidar[:, 1] / (self.real_h)) * 2
            lidar[:, 2] = ((lidar[:, 2] + 2.) / (self.real_z)) * 2 - 1
        out = self.header(input_dict, img_metas, self.openset, self.ms, lidar)
        if self.use_sem:
            out["semantic"] = sem_out_list
        if self.use_depth:
            out["depth"] = depth_out_list
        # NOTE:2023/9/4
        if self.use_large_model:
            out["img_feat"] = img_feat
        
        # save_path = os.path.join("/DATA_EDS/zyp/Surround_scene/bev_feat/sequences/", img_metas[0]['sequence_id'], "bev_feat_epoch_11", img_metas[0]['frame_id'] + ".pt")
        # save_dir = os.path.dirname(save_path)
        # os.makedirs(save_dir, exist_ok=True)
        # bev_feat = out['bev'][0].cpu()
        # torch.save(bev_feat, save_path)
        # print(save_path)
        # print(out['bev_full'][0].shape)

        # if self.fine_grain:
        #     out['occupied'] = out_occ

            # print(sem_out.,shape)
        return out 

    def step(self, out_dict, bev_feat, target, img_metas, step_type):
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
        if self.distill:
            out_bev = out_dict["bev"]
        if self.ms:
            ssc_pred_2 = out_dict["ssc_logit_2"]
            ssc_pred_4 = out_dict["ssc_logit_4"]
            ssc_pred_8 = out_dict["ssc_logit_8"]
            ssc_preds = [ssc_pred, ssc_pred_2, ssc_pred_4, ssc_pred_8]
            target_2 = torch.tensor(img_metas[0]["target_2"], device=ssc_pred.device).unsqueeze(0)
            target_4 = torch.tensor(img_metas[0]["target_4"], device=ssc_pred.device).unsqueeze(0)
            target_8 = torch.tensor(img_metas[0]["target_8"], device=ssc_pred.device).unsqueeze(0)
            targets = [target, target_2, target_4, target_8]
        
        if self.use_sem:
            sem_pred = out_dict["semantic"]
            target_2d = torch.tensor(img_metas[0]["semantic"], device=ssc_pred.device)
            # save_root = '/home/aidrive/zyp/Surround_scene/sem/pred_14.10'
            # os.makedirs(save_root, exist_ok=True)
            # npy_name = img_metas[0]["img_filename"][0].replace(".png", ".npy").replace("./kitti/dataset/sequences/08/image_2/", "")
            # save_path = os.path.join(save_root, npy_name)
            # np.save(save_path, sem_pred[0].cpu().numpy())
        
        if self.use_depth:
            depth_pred = out_dict["depth"]
            target_depth = torch.tensor(img_metas[0]["depth"], device=ssc_pred.device) 
            
        if self.use_lidar:
            lidar_out = out_dict["lidar_out"]
            target_lidar = torch.tensor(img_metas[0]["lidar_label"], device=ssc_pred.device).permute(1,0)
        
        if step_type == "train":
            loss_dict = dict()
            bce_class_weight = self.bce_class_weights.type_as(target)
            class_weight = self.class_weights.type_as(target)
            if self.use_sem:
                ce_loss = CE_loss_2D(sem_pred, target_2d, self.ratio_2D)
                loss_dict['loss_2d_ce'] = ce_loss
            
            if self.use_depth:
                depth_silog_loss = silog_loss(depth_pred, target_depth)
                loss_dict['depth_loss'] = depth_silog_loss
            
            if self.distill:
                target_2 = torch.tensor(img_metas[0]["target_2"], device=ssc_pred.device).unsqueeze(0)
                distill_loss = Distill_loss(out_bev, bev_feat.squeeze(0), target_2, self.ratio_distill)
                loss_dict['distill_loss'] = distill_loss

            if self.openset:
                openssc_pred = out_dict['openssc_logit']
                b, h, w, z = target.shape
                target_feat = torch.zeros((h, w, z, 768)).to(target.device)
                # print(self.text_features)
                for j in range(len(self.text_features)):
                    target_feat[torch.where(target[0] == j)] = self.text_features[j].detach()
                cos_loss = cos_similarity(openssc_pred, target, target_feat)
                loss_dict['loss_openset'] = cos_loss
            
            # NOTE:2023/9/4
            if self.use_large_model:
                distill_2d_pred = out_dict['img_feat'].squeeze(0)  # [1, 512, h/16, w/16]
                large_feat = torch.tensor(img_metas[0]["large_feat"], device=ssc_pred.device).squeeze(0) # [1, 512, h/4, w/4]

                distill_2d_loss = cos_similarity_2d(distill_2d_pred, large_feat, self.ratio_2d_distill)
                loss_dict['distill_2d_loss'] = distill_2d_loss


            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight, self.ratio_3D)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target, self.ratio_sem)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target, self.ratio_geo)
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

    def training_step(self, out_dict, bev_feat, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, bev_feat, target, img_metas, "train")

    def validation_step(self, out_dict, bev_feat, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, bev_feat, target, img_metas, "val")

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
