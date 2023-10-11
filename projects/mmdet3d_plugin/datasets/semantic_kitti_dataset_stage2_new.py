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
import yaml
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.voxformer.utils.ssc_metric import SSCMetrics

@DATASETS.register_module()
class SemanticKittiDatasetStage2(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        temporal = [],
        eval_range = 51.2,
        depthmodel="msnet3d",
        nsweep=10,
        labels_tag = 'labels',
        query_tag = 'query_iou5203_pre7712_rec6153',
        color_jitter=None,
    ):
        super().__init__()

        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.query_tag = query_tag
        self.nsweep=str(nsweep)
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["22"],
        }
        self.split = split 
        self.sequences = splits[split]
        with open(os.path.join(self.data_root, "dataset", "semantic-kitti.yaml"), 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.learning_map = semkittiyaml['learning_map']

        self.n_classes = 20
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.metrics = SSCMetrics(self.n_classes)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2
        self.img_W = 1220
        self.img_H = 370

        self.poses=self.load_poses()
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

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        # if sef
        calib_out["P3"] = calib_all["P3"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            # proj_matrix_2 = P2 @ T_velo_2_cam
            
            P3 = np.zeros((3, 4))
            # proj_matrix_3 = np.zeros((3, 4))
            
            if len(self.target_frames) == 1:
                P3 = calib["P3"]
                # proj_matrix_3 = P3 @ T_velo_2_cam
                
                
            glob_path = os.path.join(
                self.data_root, "dataset", "sequences_" + self.depthmodel + "_sweep"+ self.nsweep, sequence, "queries", "*." + self.query_tag
            )

            for proposal_path in glob.glob(glob_path):
            
                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proposal_path": proposal_path,
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
        scan = self.scans[index]

        proposal_path = scan["proposal_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(proposal_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id, proposal_path)
        img, bev_feat, openset_feat, sem, lidar, (w, h) = self.get_input_info(sequence, frame_id)
        target, target_2, target_4, target_8 = self.get_gt_info(sequence, frame_id)
        meta_dict["semantic"] = sem
        # meta_dict["depth"] = depth
        # meta_dict["large_feat"] = large_feat
        meta_dict['img_shape'] = [(w, h)]
        # print(self.img_H, self.img_W)
        # print(w,h)
        
        meta_dict["lidar_label"] = lidar[:,-1:]
        meta_dict["lidar"] = lidar[:,0:-1]
        
        meta_dict["target_2"] = target_2
        meta_dict["target_4"] = target_4
        meta_dict["target_8"] = target_8

        # meta_dict["proposal"] = gt_proposal.reshape(-1).astype(np.float32)
        data_info = dict(
            img_metas = meta_dict,
            img = img,
            bev_feat = bev_feat,
            openset_feat = openset_feat,
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
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P2 = scan["P2"]
        cam_k = P2[0:3, 0:3]
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = np.identity(4)
        lidar2img_rt[:3, :] = (P2 @ lidar2cam_rt)
        # lidar2img_rt = (viewpad @ lidar2cam_rt)

        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        # for reference img
        seq_len = len(self.poses[sequence])
        
        for i in self.target_frames:
            if len(self.target_frames) == 1:
                rgb_path = os.path.join(
                    self.data_root, "dataset", "sequences", sequence, "image_3", frame_id + ".png"
                )
                P3 = scan["P3"]
                cam_k = P3[0:3, 0:3]
                intrinsic = cam_k
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.identity(4)
                lidar2img_rt[:3, :] = (P3 @ lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                cam_intrinsics.append(intrinsic)
                image_paths.append(rgb_path) 
                break
                              
            elif len(self.target_frames) > 2:
                id = int(frame_id)

                if id + i < 0 or id + i > seq_len-1:
                    target_id = frame_id
                else:
                    target_id = str(id + i).zfill(6)

                rgb_path = os.path.join(
                    self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
                )

                pose_list = self.poses[sequence]

                ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
                target = pose_list[int(target_id)]
                ref2target = np.matmul(inv(target), ref) # both for lidar

                target2cam = scan["T_velo_2_cam"] # lidar to camera
                ref2cam = target2cam @ ref2target

                lidar2cam_rt  = ref2cam
                lidar2img_rt = (viewpad @ lidar2cam_rt)

                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                cam_intrinsics.append(intrinsic)
                image_paths.append(rgb_path)

        proposal_bin = self.read_occupancy_SemKITTI(proposal_path)

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

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        if self.split == "train" or self.split == "val":
            seq_len = len(self.poses[sequence])
            image_list = []
            openset_feat_list = []
            sem_list = []
            # depth_list = []
            bev_feat_list = []
            # large_feat_list = []
            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
            )
            openset_feat_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "openset", "image_2", frame_id + ".pt"
            )
            # 15.02
            # 15.46
            # 15.53
            # epoch 16: 15.03
            # epoch 11: 15.69
            bev_feat_path = os.path.join(
                "/home/aidrive/zyp/Surround_scene/bev_feat", "sequences", sequence, "bev_feat_15.02", frame_id + ".pt"
            )
            # bev_feat_path = os.path.join(
            #     self.data_root, "dataset", "sequences", sequence, "bev_feat", frame_id + ".pt"
            # )
            sem_path = os.path.join(
                "/home/aidrive/zyp/Surround_scene/sem", "sequences", sequence, frame_id + ".npy"
            )
            # sem_path = rgb_path.replace("image_2", "sem_gt")
            # depth_path = os.path.join(
            #     "/home/aidrive/zyp/Surround_scene/depth", "sequences", sequence, frame_id + ".npy"
            # )
            # large_feat_path = os.path.join(
            #     "/home/aidrive/zyp/large_feat", sequence, "image_2", frame_id + "_decoder.pt"
            # )

            
            # add lidar
            # lidar = 0
            lidar_path = rgb_path.replace("image_2", "velodyne").replace(".png", ".bin")
            label_path = rgb_path.replace("image_2", "labels").replace(".png", ".label")
            
            img = Image.open(rgb_path).convert("RGB")
            bev_feat = torch.load(bev_feat_path)
            # large_feat = torch.load(large_feat_path)
            # bev_feat = torch.zeros((1,1))
            # openset_feat = torch.load(openset_feat_path).type(torch.float32)
            openset_feat = torch.zeros((1,1))
            # w, h = img.size
            # scal_x = 1
            # scal_y = 1 
            # if w != 1241:
            #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)
            #     sem = sem.resize((self.img_W, self.img_H), Image.NEAREST)

            #     scal_x = self.img_W / w
            #     scal_y = self.img_H / h
            sem = np.load(sem_path)
            # sem = Image.open(sem_path)
            # sem = np.array(sem)
            # depth = np.load(depth_path)
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            # img = img[:self.img_H, :self.img_W, :]  # crop image
            # sem = sem[:self.img_H, :self.img_W]
            
            w, h, _ = img.shape
            
            image_list.append(self.normalize_rgb(img))
            openset_feat_list.append(openset_feat)
            sem_list.append(sem)
            # depth_list.append(depth)
            bev_feat_list.append(bev_feat)
            # large_feat_list.append(large_feat)
            
            lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape((-1, 4))
            label = np.fromfile(label_path, dtype=np.uint32, count=-1)
            label = label & 0xFFFF
            label = np.vectorize(self.learning_map.__getitem__)(label)
            label[label == 0] = 255
            lidar_mask =  (lidar[:, 0] >= 0.) & (lidar[:, 0] <= 51.2) & (lidar[:, 1] >= -25.6) & (lidar[:, 1] <= 25.6) & (lidar[:, 2] >= -2.) & (lidar[:, 2] <= 4.4)
            label = label[lidar_mask]
            lidar = lidar[lidar_mask]
            lidar[:,-1] = label
            # reference frame
            for i in self.target_frames:
                if len(self.target_frames) == 1:
                    rgb_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "image_3", frame_id + ".png"
                    )
                    openset_feat_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "openset", "image_3", frame_id + ".pt"
                    )
                    openset_feat = torch.load(openset_feat_path).type(torch.float32)
                    sem_path = os.path.join(
                        "/home/aidrive/zyp/Surround_scene/sem", "sequences", sequence, frame_id + ".npy"
                    )
                    # sem_path = rgb_path.replace("image_3", "sem_gt_3")
                    # depth_path = os.path.join(
                    #     "/DATA_EDS/zyp/Surround_scene/depth", "sequences", sequence, frame_id + ".npy"
                    # )
                    img = Image.open(rgb_path).convert("RGB")
                    # if img.size[0] == 1226:
                    #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)
                    #     sem = sem.resize((self.img_W, self.img_H), Image.NEAREST)
                    sem = np.load(sem_path)
                    # sem = Image.open(sem_path)
                    # sem = np.array(sem)
                    # depth = np.load(depth_path)
                    if self.color_jitter is not None:
                        img = self.color_jitter(img)
                    # PIL to numpy
                    img = np.array(img, dtype=np.float32, copy=False) / 255.0
                    # img = img[:self.img_H, :self.img_W, :]  # crop image
                    # sem = sem[:self.img_H, :self.img_W]

                    image_list.append(self.normalize_rgb(img))
                    openset_feat_list.append(openset_feat)
                    sem_list.append(sem)  
                    # depth_list.append(depth)
                else:
                    id = int(frame_id)

                    if id + i < 0 or id + i > seq_len-1:
                        target_id = frame_id
                    else:
                        target_id = str(id + i).zfill(6)

                    rgb_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
                    )
                    openset_feat_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "openset", "image_2", target_id + ".pt"
                    )
                    # openset_feat = torch.load(openset_feat_path).type(torch.float32)
                    openset_feat = torch.zeros((1,1))
                    # bev_feat = torch.zeros((1,1))
                    sem_path = os.path.join(
                        "/home/aidrive/zyp/Surround_scene/sem", "sequences", sequence, frame_id + ".npy"
                    )
                    # sem_path = rgb_path.replace("image_2", "sem_gt")
                    # depth_path = os.path.join(
                    #     "/DATA_EDS/zyp/Surround_scene/depth", "sequences", sequence, target_id + ".npy"
                    # )
                    img = Image.open(rgb_path).convert("RGB")
                    # w, h = img.size
                    # if w != 1241:
                    #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)
                    #     sem = sem.resize((self.img_W, self.img_H), Image.NEAREST)
                    sem = np.load(sem_path)
                    # sem = Image.open(sem_path)
                    # sem = np.array(sem)
                    # depth = np.load(depth_path)
                    # Image augmentation
                    if self.color_jitter is not None:
                        img = self.color_jitter(img)
                    # PIL to numpy
                    img = np.array(img, dtype=np.float32, copy=False) / 255.0
                    # img = img[:self.img_H, :self.img_W, :]  # crop image
                    # sem = sem[:self.img_H, :self.img_W]

                    image_list.append(self.normalize_rgb(img))
                    openset_feat_list.append(openset_feat)
                    sem_list.append(sem)
                    # depth_list.append(depth)
            image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
            bev_feat = torch.stack(bev_feat_list, dim=0)
            # large_feat = torch.stack(large_feat_list, dim=0)
            sem = np.array(sem_list)
            # depth = np.array(depth_list)
            openset_feat = torch.stack(openset_feat_list, dim=0)
        else:
            seq_len = len(self.poses[sequence])
            image_list = []

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # w, h = img.size
            # scal_x = 1
            # scal_y = 1 
            # if w != 1241:
            #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)

            #     scal_x = self.img_W / w
            #     scal_y = self.img_H / h

            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            # img = img[:self.img_H, :self.img_W, :]  # crop image
            w, h, _ = img.shape

            image_list.append(self.normalize_rgb(img))
            
            # reference frame
            for i in self.target_frames:
                if len(self.target_frames) == 1:
                    rgb_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "image_3", frame_id + ".png"
                    )
                    img = Image.open(rgb_path).convert("RGB")
                    # w, h = img.size
                    # if w != 1241:
                    #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)
                    if self.color_jitter is not None:
                        img = self.color_jitter(img)
                    # PIL to numpy
                    img = np.array(img, dtype=np.float32, copy=False) / 255.0
                    # img = img[:self.img_H, :self.img_W, :]  # crop image

                    image_list.append(self.normalize_rgb(img))         
                else:
                    id = int(frame_id)

                    if id + i < 0 or id + i > seq_len-1:
                        target_id = frame_id
                    else:
                        target_id = str(id + i).zfill(6)

                    rgb_path = os.path.join(
                        self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
                    )
                    img = Image.open(rgb_path).convert("RGB")
                    # w, h = img.size
                    # if w != 1241:
                    #     img = img.resize((self.img_W, self.img_H), Image.BICUBIC)

                    # Image augmentation
                    if self.color_jitter is not None:
                        img = self.color_jitter(img)
                    # PIL to numpy
                    img = np.array(img, dtype=np.float32, copy=False) / 255.0
                    # img = img[:self.img_H, :self.img_W, :]  # crop image

                    image_list.append(self.normalize_rgb(img))
            image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
            bev_feat = torch.ones_like(image_tensor)
            # large_feat = torch.ones_like(image_tensor)
            openset_feat = torch.ones_like(image_tensor)
            sem = np.ones((256,256,32))
            # depth = np.ones((256,256,32))
            lidar = np.ones((256,256,32))

        return image_tensor, bev_feat, openset_feat, sem, lidar, (w, h)

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val":
            # load full-range groundtruth
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            target_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_2 = np.load(target_2_path)
            target_4_path = os.path.join(self.label_root, sequence, frame_id + "_1_4.npy")
            target_4 = np.load(target_4_path)
            target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
            target_8 = np.load(target_8_path)
            # target_2[np.logical_and(target_2 > 0, target_2 < 255)] = 1
            # target_2[target_2 == 255] = 0
            # short-range groundtruth
            if self.eval_range == 25.6:
                target[128:, :, :] = 255
                target[:, :64, :] = 255
                target[:, 192:, :] = 255

            elif self.eval_range == 12.8:
                target[64:, :, :] = 255
                target[:, :96, :] = 255
                target[:, 160:, :] = 255
        else:
            target = np.ones((256,256,32))
            target_2 = np.ones((256,256,32))
            target_4 = np.ones((256,256,32))
            target_8 = np.ones((256,256,32))

        return target, target_2, target_4, target_8

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
