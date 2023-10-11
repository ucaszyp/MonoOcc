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
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import yaml
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as mask_util

save_path = '/home/aidrive/zyp/Surround_scene/fisherocc/result/MonoOcc-demo-12.35'
voxel_path = '/home/aidrive/zyp/Surround_scene/fisherocc/kitti/dataset'

# inverse of previous map
remapdict = {
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81    # "traffic-sign"
}

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """

    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def pack(array):
    """ convert a boolean array into a bitwise array. """
    array = array.reshape((-1))
    compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    return np.array(compressed, dtype=np.uint8)

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    nr_classes = len(remapdict)

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    config_path = "kitti/dataset/semantic-kitti.yaml"
    dataset_config = yaml.safe_load(open(config_path, 'r'))
    inv_map = np.zeros(20, dtype=np.int32)
    inv_map[list(dataset_config['learning_map_inv'].keys())] = list(dataset_config['learning_map_inv'].values())

    model.eval()
    results = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    # have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            # print(result)
            # encode mask results
            if isinstance(result, dict):
                # if 'y_pred' in result.keys():
                # y_pred = result['y_pred']
                batch_size = len(result['y_pred'])
                # y_preds.extend(y_pred)

                if dataset.split == "test":
                    # img_filename = result['img_filename']
                    # img_path = img_filename.replace("./kitti/dataset", save_path).replace("/image_2", "/predictions").replace(".png", ".label")
                    # prediction = result['y_pred']

                    # dir_name = osp.dirname(img_path)
                    # os.makedirs(dir_name, exist_ok=True)

                    # prediction = prediction.reshape((-1)).astype(np.uint32)
                    # prediction = remap_lut[prediction]
                    # prediction = prediction.astype(np.uint16)

                    # bin_path = img_path.replace(save_path, voxel_path).replace("/predictions", "/voxels").replace(".label", ".bin")
                    # if osp.exists(bin_path):
                    #     prediction.tofile(img_path)

                    # for validation, generate both the occupancy & input image for visualization
                    prediction = result['y_pred']
                    output_voxels = prediction.reshape(-1)
                    output_voxels = output_voxels.astype(np.uint8)

                    img_filename = result['img_filename']
                    raw_img = np.array(Image.open(img_filename))

                    img_path = img_filename.replace("./kitti/dataset", save_path).replace("/image_2", "/predictions").replace(".png", ".pkl")

                    out_dict = dict(
                        output_voxel=output_voxels,
                        raw_img=raw_img,
                    )

                    dir_name = osp.dirname(img_path)
                    os.makedirs(dir_name, exist_ok=True)
                    
                    with open(img_path, "wb") as handle:
                        pickle.dump(out_dict, handle)
                        # print("wrote to", img_path)
                
                # if dataset.split == "val":
                #     # for validation, generate both the occupancy & input image for visualization
                #     prediction = result['y_pred']
                #     output_voxels = prediction.reshape(-1)
                #     output_voxels = output_voxels.astype(np.uint8)

                #     img_filename = result['img_filename']
                #     raw_img = np.array(Image.open(img_filename))

                #     img_path = img_filename.replace("./kitti/dataset", save_path).replace("/image_2", "/predictions").replace(".png", ".pkl")

                #     out_dict = dict(
                #         output_voxel=output_voxels,
                #         raw_img=raw_img,
                #     )

                #     dir_name = osp.dirname(img_path)
                #     os.makedirs(dir_name, exist_ok=True)
                    
                #     with open(img_path, "wb") as handle:
                #         pickle.dump(out_dict, handle)
                #         # print("wrote to", img_path)

                # y_true = result['y_true']
                # batch_size = len(result['y_true'])
                results.append(result)
                # if 'mask_results' in result.keys() and result['mask_results'] is not None:
                #     mask_result = custom_encode_mask_results(result['mask_results'])
                #     mask_results.extend(mask_result)
                #     have_mask = True
            # else:
            #     batch_size = len(result)
            #     bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        # if have_mask:
        #     mask_results = collect_results_gpu(mask_results, len(dataset))
        # else:
        #     mask_results = None
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        # tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        # if have_mask:
        #     mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        # else:
        #     mask_results = None

    # if mask_results is None:
    return results
    # return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
