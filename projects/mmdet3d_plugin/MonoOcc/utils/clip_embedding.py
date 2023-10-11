import torch
import clip

NUSCENES_LABELS_DETAILS = ('barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
                           'motorcycle', 'person', 'pedestrian','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                           'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                           'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods')
NUSCENES_LABELS_16 = ('barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person', 'traffic cone',
                      'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
KITTI_LABELS_19 = ("empty space", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                   "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign")

def obtain_text_features_and_palette(dataset, extractor):
    '''obtain the CLIP text feature and palette.'''
    labelset = []
    if dataset == 'nyu':
        labelset = list(NYU_LABELS_20)
        labelset[-1] = 'other'
      
    elif dataset == 'nuscenes':
        labelset = list(NUSCENES_LABELS_16)
        
    elif dataset == "KITTI":
        labelset = list(KITTI_LABELS_19)

    if extractor == 'openseg':
        model_name="ViT-L/14@336px"
        postfix = '_768' # the dimension of CLIP features is 768
    elif extractor == 'lseg':
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    text_features = extract_clip_feature(labelset, model_name=model_name)

    return text_features

def extract_text_feature(labelset, extractor):
    '''extract CLIP text features.'''

    # a bit of prompt engineering

    labelset = [ "a " + label + " in a scene" for label in labelset]
    if extractor == 'lseg':
        text_features = extract_clip_feature(labelset)
    elif extractor == 'openseg':
        text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")
    else:
        raise NotImplementedError

    return text_features

def extract_clip_feature(labelset, model_name="ViT-B/32"):
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features
# def valid_code(name, token, points, lidar_seg, prop_lidar_openseg_feat, lidar_openseg_feat_mask, text_features, labelset, mapper):
    
#     points = points.transpose(1,0)[:,:3]
#     text_features = text_features.cuda()
    
#     ### gt label
#     gt_label = remap_lut[lidar_seg]-1
#     gt_label[gt_label==-1] = 16
#     # !!!!!!!!!!
#     # gt_label[gt_label!=18] = 0
#     gt_label[gt_label!=10] = 0
#     # gt_label[(gt_label != 10) * (gt_label != 12)] = 0
#     gt_label[(gt_label==10)] = 1
#     # gt_label[gt_label==12] = 1
#     gt_label_color = convert_labels_with_palette(gt_label, palette)

#     ### pred label
#     prop_lidar_openseg_feat_tensor = torch.tensor(prop_lidar_openseg_feat).half().cuda()
#     dot_result = prop_lidar_openseg_feat_tensor @ text_features.t()
#     dot_result_logits = F.sigmoid(dot_result)
    
#     logits_pred = ((dot_result_logits[:,-1] > 0.5) * (dot_result_logits[:,-1] <= 1)).detach().cpu()
#     logits_pred = np.array(logits_pred)
#     pred_label_color = convert_labels_with_palette(logits_pred, palette)
#     # pred_label_color[~lidar_openseg_feat_mask] = 1 # 看哪些是扩散了feat的
#     point_pred_label = np.concatenate([points, pred_label_color*255], axis=1)


# TODO: change nuscenes to kitti

# def precompute_text_related_properties():
#     '''pre-compute text features, labelset, palette, and mapper.'''
#     labelset = list(NUSCENES_LABELS_16)

#     map_nuscenes_details = False
#     mapper = None
#     if map_nuscenes_details:
#         labelset = list(NUSCENES_LABELS_DETAILS)
#         mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)
#     text_features = extract_text_feature(labelset)

#     return text_features, labelset, mapper