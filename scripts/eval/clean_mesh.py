import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh
import argparse
from pathlib import Path

def scale_mesh(mesh, scan):
    
    # load offset and scale factor
    data_dir = os.path.join('../data/DTU_NeuS', f'dtu_scan{scan}')
    cam_file = f'{data_dir}/cameras_sphere.npz'
    camera_dict = np.load(cam_file)
    scale_mat = camera_dict['scale_mat_0'].astype(np.float32)

    # transform vertices to world 
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]

    return mesh

def clean_points_by_mask(points, scan):
    cameras = np.load('../data/DTU_NeuS/dtu_scan{}/cameras_sphere.npz'.format(scan))
    mask_lis = sorted(glob('../data/DTU_NeuS/dtu_scan{}/mask/*.png'.format(scan)))
    n_images = 49 if scan < 83 else 64
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(n_images):
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (101, 101))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask

def clean_mesh(old_mesh_path, scan, need_scale=False, need_clean=True):
    old_mesh = trimesh.load(old_mesh_path)
    if need_scale:
        old_mesh = scale_mesh(old_mesh, scan)
    if need_clean:
        old_vertices = old_mesh.vertices[:]
        old_faces = old_mesh.faces[:]
        mask = clean_points_by_mask(old_vertices, scan)
        indexes = np.ones(len(old_vertices)) * -1
        indexes = indexes.astype(np.int64)
        indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

        faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
        new_faces = old_faces[np.where(faces_mask)]
        new_faces[:, 0] = indexes[new_faces[:, 0]]
        new_faces[:, 1] = indexes[new_faces[:, 1]]
        new_faces[:, 2] = indexes[new_faces[:, 2]]
        new_vertices = old_vertices[np.where(mask)]

        new_mesh = trimesh.Trimesh(new_vertices, new_faces)
        
        meshes = new_mesh.split(only_watertight=False)
        new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    else:
        new_mesh = old_mesh
    return new_mesh

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--scan', default='None', type=str)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()
    root_dir = args.exp_dir
    new_dir = root_dir if args.override else os.path.join(root_dir, 'cleaned')
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    
    if args.scan == 'None':
        scans = [ 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122 ]
    else:
        scans = [int(args.scan)]
        
    for scan in scans:
        old_mesh_path = os.path.join(root_dir, f'scan{scan}.ply')
        new_mesh = clean_mesh(old_mesh_path, scan)
        new_mesh.export(os.path.join(new_dir, f'scan{scan}.ply'))
        print('dtu_scan',str(scan),' has been cleaned!')
    