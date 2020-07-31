import numpy as np

def get_xyz(depth_image):
    intrin = {
        'ppx': 317.241, 'ppy': 242.779, 'fx': 385.337, 'fy': 385.337
    }
    def deproject(intrin, pixel, depth):
        x = (pixel[0] - intrin['ppx']) / intrin['fx']
        y = (pixel[1] - intrin['ppy']) / intrin['fy']
        return depth * x, depth * y, depth
    pix2xyz = lambda pixel, depth : deproject(intrin, pixel, depth)
    xyz_matrix = []
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i][j] != 0:
                depth = depth_image[i][j]
                pixel = (i, j)
                xyz = pix2xyz(pixel, depth)
                xyz_matrix.append(xyz)
                
                
    return xyz_matrix

def xy_heatmap(ppl_depth, gnd_depth):
    xyz_gnd = np.array(get_xyz(gnd_depth))
    xyz_gnd = xyz_gnd[np.random.choice(xyz_gnd.shape[0], size=300, replace=False), :]
    xyz_ppl = np.array(get_xyz(ppl_depth))
    xyz_ppl = xyz_ppl[np.argsort(-1 * xyz_ppl[:, 0]), :][:100, :]
    xyz_ppl_mean = xyz_ppl.mean(axis=0)
    xyz_ppl = np.random.multivariate_normal(
        xyz_ppl_mean, [[200, 0, 0], [0, 200, 0], [0, 0, 200]], 100
    )
    
    floor_centered = xyz_gnd - xyz_gnd.mean(axis=0)
    ppl_flr_center = xyz_ppl - xyz_gnd.mean(axis=0)
    
    floor_sample = floor_centered[
        np.random.choice(floor_centered.shape[0], size=300, replace=False), :
    ]
    _, _, vt = np.linalg.svd(floor_sample)
    norm = vt[2]
    
    gnd_dists = ppl_flr_center @ norm
    ppl_projs = np.repeat(
        norm[np.newaxis,...], ppl_flr_center.shape[0], axis=0
    ) * gnd_dists[...,np.newaxis]
    ppl_projs = ppl_flr_center - ppl_projs
    
    gnd_dists2 = floor_centered @ norm
    gnd_projs = np.repeat(
        norm[np.newaxis,...], floor_centered.shape[0], axis=0
    ) * gnd_dists2[...,np.newaxis]
    gnd_projs = floor_centered - gnd_projs
    
    trans_ppl_pts = ppl_projs + xyz_gnd.mean(axis=0)
    trans_gnd_pts = gnd_projs + xyz_gnd.mean(axis=0)
    all_proj_pts = np.concatenate([trans_gnd_pts, trans_ppl_pts], axis=0)
    _, _, vt2 = np.linalg.svd(all_proj_pts - all_proj_pts.mean(axis=0))
    
    unitcross = lambda a, b : np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    no_xy = np.array(vt2[2], copy=True)
    no_xy[0] = 0
    no_xy[1] = 0
    x, y, z = unitcross(vt2[2], no_xy)

    nvt, nx = np.linalg.norm(vt2[2]), np.linalg.norm(no_xy)
    cos_ang = np.dot(vt2[2], no_xy) / (nvt * nx)
    
    c = cos_ang
    s = np.sqrt(1-c*c)
    C = 1-c
    rmat = np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s ],
                     [y*x*C+z*s, y*y*C+c, y*z*C-x*s ],
                     [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])

    xyz = all_proj_pts - all_proj_pts.mean(axis=0)
    rot_xyz = rmat @ (xyz.T)
    rot_xyz = rot_xyz.T
    rot_xyz[:, 2] = 0
    return rot_xyz, trans_gnd_pts.shape[0]