"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import io_data as SemanticKittiIO
import argparse
import yaml

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()
        # count for empty
        zero_count_0 = np.array(np.where(label_bin == 17)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 17 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin >= 0, label_bin < 17))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def main(config):
    scene_size = (200, 200, 16)
    sequences = sorted(os.listdir(os.path.join(config.nusc_root, "trainval", "gts")))
    for i in tqdm(range(150, 250)):
        sequence = sequences[i]
        sequence_path = os.path.join(config.nusc_root, "trainval", "gts", sequence)
        sample_list = sorted(os.listdir(sequence_path))
        out_dir = os.path.join(config.nusc_preprocess_root, "trainval", "gts", sequence)
        os.makedirs(out_dir, exist_ok=True)

        downscaling = {"1_2": 2}

        for i in tqdm(range(len(sample_list))):
            sample_path = os.path.join(sequence_path, sample_list[i], "labels.npz")
            label_data = np.load(sample_path)
            LABEL = label_data['semantics']
            mask = label_data['mask_camera']
            LABEL[mask == 0] = 255
            # LABEL[LABEL == 17] = 0
            #todo: get label from data root

            for scale in downscaling:
                filename = sample_path.replace("labels.npz", "labels_" + scale + "_neww.npy")
                # print(filename)
                # label_filename = os.path.join(out_dir, filename)
                # If files have not been created...
                # print(filename)
                if not os.path.exists(filename):
                    if scale == "1_2":
                        LABEL_ds = _downsample_label(
                            LABEL, (200, 200, 16), downscaling[scale]
                        )
                    else:
                        LABEL_ds = LABEL
                    np.save(filename, LABEL_ds)
                    # print("wrote to", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./label_preprocess.py")
    parser.add_argument(
        '--nusc_root',
        '-r',
        type=str,
        help='nusc_root',
        default='nuscenes'
    )

    parser.add_argument(
        '--nusc_preprocess_root',
        '-p',
        type=str,
        help='nusc_preprocess_root',
        default='nuscenes'
    )
    config, unparsed = parser.parse_known_args()
    main(config)
