# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MonoOcc/blob/main/LICENSE

import os
import glob
import copy
import torch
import random
import mmcv
import numpy as np
from torch.utils.data import Dataset
from os import path as osp
from PIL import Image
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.MonoOcc.utils.ssc_metric import SSCMetrics
import yaml
import pickle
from nuscenes import NuScenes

@DATASETS.register_module()
class nuScenesDatasetStage1(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        depthmodel="sdnnet",
        nsweep=5,
    ):
        super().__init__()
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
        self.split = split
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, "labels")
        self.n_classes = 2
        self.label_mapping = os.path.join(data_root, "nuscenes.yaml")
        self.train_imagest = os.path.join(data_root, "trainval", "info", "nuscenes_infos_train.pkl")
        self.val_imagest = os.path.join(data_root, "trainval", "info", "nuscenes_infos_val.pkl")
        
        if split == "val" or split == "test":
            with open(self.val_imagest, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.train_imagest, 'rb') as f:
                data = pickle.load(f)
        self.data_info = data["infos"]
        
        # with open(self.label_mapping, 'r') as stream:
        #     nuscenesyaml = yaml.safe_load(stream)
        self.nsweep=str(nsweep)
        self.depthmodel = depthmodel
        self.class_names =  ["empty", "occupied"]
        self.metrics = SSCMetrics(2)

        self.voxel_size = 0.4
        self.img_W = 1600
        self.img_H = 896
        
        self.load_scans()
        self.test_mode = test_mode
        self._set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)


    def load_scans(self):
        self.scans = []
        for info in self.data_info:
            sample_token = info["token"]
            scene = info["scene"]
            pseudo_path = os.path.join(self.data_root, "trainval", "nuscenes-voxel-midas-sweep0", scene, sample_token + ".npy")
            self.scans.append(
                {
                    "sequence": scene,
                    "frame_id": sample_token,
                    "voxel_path": pseudo_path,
                }
            )
        self.scans = self.scans


    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. 
        """
        scan = self.scans[index]
        scene = scan["sequence"]
        sample_token = scan["frame_id"]
        voxel_path = scan["voxel_path"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(
            self.data_root, "trainval", scene, sample_token + ".png"
        )
        image_paths = []
        image_paths.append(rgb_path)

        # load voxelized pseudo point cloud
        pseudo_pc_bin = np.load(voxel_path)[0].astype(np.float32)
        pseudo_pc_bin = pseudo_pc_bin.transpose(1, 0, 2)
        pseudo_pc_bin = pseudo_pc_bin[:, ::-1, :]
        
        if self.split == "train" or self.split == "val":
            # load ground truth
            targets = []
            if self.split == "train":
                target_1_2_path = os.path.join(self.data_root, "trainval", "gts", scene, sample_token, "labels_1_2.npy")
            else:
                target_1_2_path = os.path.join(self.data_root, "trainval", "gts", scene, sample_token, "labels_1_2_new.npy")
            target = np.load(target_1_2_path)
            target = target.reshape(-1)
            target = target.reshape(100, 100, 8)
            target = target.astype(np.float32)
            target[target == 17] = 0
            targets.append(target)  
            
            target_1_4_path = os.path.join(self.data_root, "trainval", "gts", scene, sample_token, "labels_1_4.npy")
            target = np.load(target_1_4_path)
            target = target.reshape(-1)
            target = target.reshape(50, 50, 4)
            target = target.astype(np.float32)
            target[target == 17] = 0    
            targets.append(target)  
             
            target_1_8_path = os.path.join(self.data_root, "trainval", "gts", scene, sample_token, "labels_1_8.npy")
            target = np.load(target_1_8_path)
            target = target.reshape(-1)
            target = target.reshape(25, 25, 2)
            target = target.astype(np.float32)
            target[target == 17] = 0   
            targets.append(target)       
            # target_1_path = target_1_2_path.replace("labels_1_2.npy", "labels.npz")
            # target_data = np.load(target_1_path)
            # mask = target_data['mask_camera']
            # target[mask == 0] = 255

        else:
            target = np.ones((100, 100, 8))

        meta_dict = dict(
            target=targets,
            sequence_id = scene,
            voxel=None,
            pseudo_pc=pseudo_pc_bin,
            img_filename=image_paths,
            img_shape = [(896, 1600)]
        )

        data_info = dict(
            img_metas = meta_dict,
            img = torch.zeros((1, 3, 896, 1600)),
            target = meta_dict['target']
        )
 
        return data_info
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        detail = dict()

        for result in results:
            self.metrics.add_batch(result['y_pred'], result['y_true'])
        metric_prefix = f'{result_name}_SemanticKITTI'

        stats = self.metrics.get_stats()
        for i, class_name in enumerate(self.class_names):
            detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

        detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
        detail["{}/IoU".format(metric_prefix)] = stats["iou"]
        detail["{}/Precision".format(metric_prefix)] = stats["precision"]
        detail["{}/Recall".format(metric_prefix)] = stats["recall"]

        self.metrics.reset()

        return detail
