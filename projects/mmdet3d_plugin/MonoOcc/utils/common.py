def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1, z_max=0.5, z_min=-0.5):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    # ipdb.set_trace()
    # pcd_show = make_o3d_pcd(p[0].detach().cpu().numpy())
    
    p_nor = p / (1 + padding + 10e-8) # (-0.5, 0.5)
    #p_nor = p_nor + 0.5 # range (0, 1)
    p_nor[:, :, 0:2] = p_nor[:, :, 0:2] + 0.5
    p_nor[:, :, 2] = (p_nor[:, :, 2] - z_min) / (z_max - z_min + 10e-8)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-8
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    
    # pcd_show2 = make_o3d_pcd(p_nor[0].detach().cpu().numpy())
    # o3d.visualization.draw_geometries([pcd_show, pcd_show2])
    
    return p_nor
