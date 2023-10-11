# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from os import path as osp
from PIL import Image
import glob
import random
import copy
import mmcv
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.MonoOcc.utils.ssc_metric import SSCMetrics
import yaml
import pickle
from nuscenes import NuScenes

@DATASETS.register_module()
class nuScenesDatasetStage2(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        temporal = [],
        eval_range = 40.0,
        depthmodel="msnet3d",
        nsweep=10,
        labels_tag = 'labels',
        query_tag = 'query_iouxxx_prexxxx_recxxxx',
        color_jitter=None,
    ):
        super().__init__()
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.query_tag = query_tag
        self.nsweep=str(nsweep)
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        self.split = split
        self.label_mapping = os.path.join(data_root, "nuscenes.yaml")
        self.train_imagest = os.path.join(data_root, "trainval", "info", "nuscenes_infos_train.pkl")
        self.val_imagest = os.path.join(data_root, "trainval", "info", "nuscenes_infos_val.pkl")
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_root + "trainval/raw_data/", verbose=True)

        if split == "val":
            with open(self.val_imagest, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.train_imagest, 'rb') as f:
                data = pickle.load(f)
        self.data_info = data["infos"]
        
        # TODO: check nuScenes classes 
        self.n_classes = 17
        self.class_names =  [ "empty", "barrier", "bicycle", "bus", "car", "construction_vehicle", 
                             "motorcycle", "pedestrian", "traffic_cone", "trailer", 
                            "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", 
                            "manmade", "vegetation"]
        self.metrics = SSCMetrics(self.n_classes)
        self.scene_size = (40.0, 40.0, 6.4)
        self.vox_origin = np.array([-20.0, -20.0, -2])
        self.voxel_size = 0.4  # 0.2m

        self.img_W = 1600
        self.img_H = 896

        # self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        
        for info in self.data_info:
            scene = info["scene"]
            sample_token = info["token"]
            # sample = self.nusc.get("sample", sample_token)
            # scene_token = sample["scene_token"]
            # scene = self.nusc.get("scene", scene_token)["name"]
            proposal_path = os.path.join(self.data_root, "trainval", "dataset", "midas", scene, sample_token + ".npy")
            img_info = self.get_info(info)
            cam_intrinsic = img_info["cam_intrinsic"]
            lidar2cam = img_info["lidar2cam"]
            lidar2img = img_info["lidar2img"]
            img_filename = img_info['img_filename']

            self.scans.append(
                {
                    "sequence": scene,
                    "frame_id": sample_token,
                    "img_filename": img_filename,
                    "lidar2img": lidar2img,
                    "lidar2cam": lidar2cam,
                    "proposal_path": proposal_path,
                    "cam_intrinsic": cam_intrinsic,
                }
            )        

    def set_group_flag(self):
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
        info = self.data_info[index]
        scan = self.scans[index]

        proposal_path = scan["proposal_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(proposal_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id, proposal_path)
        img, sem = self.get_input_info(scan)
        target = self.get_gt_info(scan)
        meta_dict["semantic"] = sem
        data_info = dict(
            img_metas = meta_dict,
            img = img,
            target = target
        )
        return data_info

    def get_meta_info(self, scan, sequence, frame_id, proposal_path):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            proposal_path (str): proposal path.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        lidar2cam_rts = scan["lidar2cam"]
        lidar2img_rts = scan["lidar2img"]
        image_paths = scan["img_filename"]
        cam_intrinsics = scan["cam_intrinsic"]
        proposal_path = scan["proposal_path"]
        # proposal_bin = self.read_occupancy_SemKITTI(proposal_path)
        proposal_bin = np.load(proposal_path).reshape(-1).astype(np.float32)
        proposal_bin[proposal_bin != 17] = 1
        proposal_bin[proposal_bin == 17] = 0
        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            proposal=proposal_bin,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            img_shape = [(self.img_H,self.img_W)]
        )

        return meta_dict

    def get_input_info(self, scan):
        """Get the image of the specific frame in a sequence.

        Args:
            scan.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.scans[0]["img_filename"])
        image_list = []
        sem_list = []
        rgb_path = scan["img_filename"][0].replace("./data/nuscenes/", "nuscenes/trainval/")
        img = Image.open(rgb_path).convert("RGB")
        img = img.resize((self.img_W, self.img_H), Image.BILINEAR)
        sem_path = rgb_path.replace("samples", "semantic/samples")
        sem_path = sem_path.replace("jpg", "png")
        sem = Image.open(sem_path)
        sem = np.array(sem)
        sem[sem == 0] = 255
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        # img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))
        sem_list.append(sem)
        # reference frame
        for i in range(1, seq_len):

            rgb_path = scan["img_filename"][i].replace("./data/nuscenes/", "nuscenes/trainval/")
            sem_path = rgb_path.replace("samples", "semantic/samples")
            sem_path = sem_path.replace("jpg", "png")
            sem = Image.open(sem_path)
            sem = np.array(sem)
            sem[sem == 0] = 255
            img = Image.open(rgb_path).convert("RGB")
            img = img.resize((self.img_W, self.img_H), Image.BILINEAR)
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            # img = img[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img))
            sem_list.append(sem)

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
        sem = np.array(sem_list)

        return image_tensor, sem

    def get_gt_info(self, scan):
        """Get the ground truth.

        Args:
            scan ,

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val" or self.split == "test":
            # load full-range groundtruth
            target_1_path = scan["proposal_path"].replace(".npy", "/labels.npz").replace("dataset/midas", "gts")
            target_data = np.load(target_1_path)
            target = target_data["semantics"].astype(np.float32)
            mask = target_data['mask_camera']
            target[target == 0] = 255
            # if self.split == "val":
            target[mask == 0] = 255    
            target[target == 17] = 0 
            # short-range groundtruth
            if self.eval_range == 20.0:
                target[:50, :, :] = 255
                target[150:, :, :] = 255
                target[:, :50, :] = 255
                target[:, 150:, :] = 255

            elif self.eval_range == 10.0:
                target[:75, :, :] = 255
                target[125:, :, :] = 255
                target[:, :75, :] = 255
                target[:, 125:, :] = 255
        else:
            target = np.ones((200,200,16))

        return target

    def read_SemKITTI(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = self.unpack(bin)
        return bin

    def read_occupancy_SemKITTI(self, path):
        occupancy = self.read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
        return occupancy

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        return np.array(compressed, dtype=np.uint8)

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

    def get_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
        """
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict