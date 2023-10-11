import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ratio):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    ) * ratio

def precision_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
    )

def sem_scal_loss(pred, ssc_target, ratio):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                

                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return ratio * loss / count

def CE_ssc_loss(pred, target, class_weights, ratio):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean * ratio

def CE_lidar_loss(pred, target, class_weights):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean

def CE_loss_2D(pred_list, target, ratio):

    criterion = nn.CrossEntropyLoss(
        ignore_index=255, reduction="none"
    )
    loss_valid_mean = 0
    N, h, w = target.shape
    if len(pred_list) > 1:
        for i in range(len(pred_list)):
            pred = pred_list[i]
            B, N, C, H, W = pred.shape
            target = target.view(B, N, h, w)
            target1 = nn.functional.interpolate(target, (H, W), mode='nearest')
            # softmax = nn.Softmax(dim=2)
            # pred = softmax(pred)
            pred = pred.view(B*N, C, H, W)
            target1 = target1.view(B*N, H, W)
            loss = criterion(pred, target1.long())
            loss_valid = loss[target1!=255]
            loss_valid_mean += (0.9 ** i) * torch.mean(loss_valid)
            
        return loss_valid_mean * ratio / 2
    else:
        pred = pred_list[0]
        B, N, C, H, W = pred.shape
        target = target.view(B, N, h, w)
        target1 = nn.functional.interpolate(target, (H, W), mode='nearest')
        # softmax = nn.Softmax(dim=2)
        # pred = softmax(pred)
        pred = pred.view(B*N, C, H, W)
        target1 = target1.view(B*N, H, W)
        loss = criterion(pred, target1.long())
        loss_valid = loss[target1!=255]
        loss_valid_mean = torch.mean(loss_valid)
        return loss_valid * ratio
    
def silog_loss(depth_est_list, depth_gt_list):
    if len(depth_est_list) > 1:
        loss_mean = 0
        for i in range(depth_est_list):
            depth_est = depth_est_list[i]
            depth_gt = depth_gt_list[i]

            B, N, C, H, W = depth_est.shape
            h, w = depth_gt.shape
            depth_gt = depth_gt.view(B, N, h, w)
            depth_gt1 = nn.functional.interpolate(depth_gt, (H, W), mode='nearest')
            depth_est = depth_est.view(B*N, C, H, W)
            depth_est = depth_est.clip(max=80., min=1e-6)
            depth_gt1 = depth_gt1.view(B*N, C, H, W)

            mask = (depth_gt1 > 1.)*(depth_gt1 < 80.)

            d = torch.log(depth_est[mask]) - torch.log(depth_gt1[mask])
            loss = torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2)) * 10.0
            loss_mean += (0.9 ** i) * loss
        return loss_mean
    else:
        depth_est = depth_est_list[0]
        depth_gt = depth_gt_list[0]

        B, N, C, H, W = depth_est.shape
        h, w = depth_gt.shape
        depth_gt = depth_gt.view(B, N, h, w)
        depth_gt1 = nn.functional.interpolate(depth_gt, (H, W), mode='nearest')
        depth_est = depth_est.view(B*N, C, H, W)
        depth_est = depth_est.clip(max=80., min=1e-6)
        depth_gt1 = depth_gt1.view(B*N, C, H, W)

        mask = (depth_gt1 > 1e-6)*(depth_gt1 < 80.)

        d = torch.log(depth_est[mask]) - torch.log(depth_gt1[mask])
        loss = torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2)) * 10.0
        return loss

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean

def cos_similarity(pred, target, target_feat):
    criterion = nn.CosineSimilarity(dim=1)
    # target = targets[0]
    # pred = preds[0]
    target_feat = target_feat.permute(3,0,1,2).unsqueeze(0)
    cos_similarity = criterion(pred, target_feat)
    loss_valid = cos_similarity[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    loss = 1 - loss_valid_mean

    return loss

def Distill_loss(bev_s, bev_t, target, ratio):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    valid_mask = target != 255
    valid = (valid_mask * nonzeros).squeeze(0)
    bev_s = bev_s.squeeze()[:,valid]
    bev_t = bev_t.squeeze()[:,valid]
    # loss = nn.MSELoss()(bev_s.unsqueeze(0), bev_t.unsqueeze(0))
    loss = nn.KLDivLoss(reduction="mean",log_target=True)(bev_s.unsqueeze(0), bev_t.unsqueeze(0))   
    return loss * ratio

def cos_similarity_2d(img_feat, large_feat, ratio):
    _, c, pred_h, pred_w = img_feat.shape
    _, c, distill_h, distill_w = large_feat.shape
    large_feat = nn.functional.interpolate(large_feat, (pred_h, pred_w), mode='bilinear', align_corners=True)

    # loss = nn.MSELoss()(img_feat.squeeze(0), large_feat.squeeze(0))
    
    criterion = nn.CosineSimilarity(dim=1)
    img_feat = img_feat.permute(0,2,3,1).reshape(-1, c)
    large_feat = large_feat.permute(0,2,3,1).reshape(-1, c)
    cos_similarity = criterion(img_feat, large_feat)
    loss_valid_mean = torch.mean(cos_similarity)
    loss = 1 - loss_valid_mean

    # img_feat = img_feat.permute(0,2,3,1).reshape(-1, c)
    # large_feat = large_feat.permute(0,2,3,1).reshape(-1, c)
    # img_feat = F.log_softmax(img_feat, dim=1)
    # large_feat = F.softmax(large_feat, dim=1)
    # loss = nn.KLDivLoss(reduction="mean")(img_feat, large_feat)

    return loss * ratio

# def BCE_ssc_loss(pred, target, class_weights, alpha=None, pred_sigma=None):
#     if alpha != None:
#         class_weights[0] = 1-alpha    # empty                 
#         class_weights[1] = alpha    # occupied                      
#     target[(target > 0) * (target < 255)] = 1
#     target = torch.tensor(target, device=pred.device).unsqueeze(0)
#     criterion = nn.CrossEntropyLoss(
#         weight=class_weights, ignore_index=255, reduction="none"
#     )
#     if pred_sigma != None:
#         # pred_sigma = pred_sigma
#     # loss_pixelwise = pred_sigma + torch.mul(torch.exp(-pred_sigma), torch.square(pred_depth - gt))
#         loss = pred_sigma + torch.mul(torch.exp(-pred_sigma), criterion(pred, target.long()))
#     else:
#         loss = criterion(pred, target.long())
#     # loss_map = (10 * criterion(pred, target.long()))[0].detach().cpu().numpy()
#     # segma_vis = pred_sigma[0].detach().cpu().numpy()
#     # np.save("1.npy",loss_map)
#     # np.save("2.npy", segma_vis)
#     loss_valid = loss[target!=255]
#     loss_valid_mean = torch.mean(loss_valid)

#     return loss_valid_mean * 2
