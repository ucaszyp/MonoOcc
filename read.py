import torch
import os
import time
from tqdm import tqdm
import numpy as np
import pickle
import torch.nn.functional as F

seq_list = ["05"]
count = 0
total_time = 0
for seq in seq_list:
    feat_root = "/DATA_EDS2/zyp/data_semantic-kitty/multiview_openseg/{}/image_2".format(seq)
    save_root = feat_root.replace("image_2", "image_2_process")
    os.makedirs(save_root, exist_ok=True)
    feat_list = sorted(os.listdir(feat_root))[::-1]
    for i in tqdm(range(len(feat_list))):
        if i % 5 == 0:
            data = {}
            feat_file = os.path.join(feat_root, feat_list[i])
            start = time.time()
            feat_file_16 = feat_file.replace("image_2", "image_2_process").replace(".pt", "_16.pt")
            feat_file_8 = feat_file_16.replace("_16.pt", "_8.pt")
            feat_file_4 = feat_file_16.replace("_16.pt", "_4.pt")
            feat_file_2 = feat_file_16.replace("_16.pt", "_2.pt")
            if os.path.exists(feat_file_16):
                count += 1
                continue
            print(feat_file_16)
            a = torch.load(feat_file)
            end = time.time()
            b = a[:, :370, :1220].float().unsqueeze(0)
            # b = a
            b_16 = F.interpolate(b, (24, 77), mode='bilinear', align_corners=True)
            b_8 = F.interpolate(b, (47, 153), mode='bilinear', align_corners=True)
            b_4 = F.interpolate(b, (93, 305), mode='bilinear', align_corners=True)
            b_2 = F.interpolate(b, (185, 610), mode='bilinear', align_corners=True)
            print(feat_file_8)
            print(feat_file_4)
            print(feat_file_2)
            b_16 = b_16.squeeze(0).half()
            b_8 = b_8.squeeze(0).half()
            b_4 = b_4.squeeze(0).half()
            b_2 = b_2.squeeze(0).half()
            print(b_16.shape)
            print(b_8.shape)
            print(b_4.shape)
            print(b_2.shape)
            torch.save(b_16, feat_file_16)
            torch.save(b_8, feat_file_8)
            torch.save(b_4, feat_file_4)
            torch.save(b_2, feat_file_2)
            count += 1
        # data['feat'] = b.numpy()
        # print(b.shape)
        # pickle_path = feat_file.replace(".pt", ".pkl").replace("image_2", "image_2_pkl")
        # with open(pickle_path, "wb") as f:
            # pickle.dump(data, f)
            total_time += end - start
print(total_time)
print(count)
print(total_time / count)