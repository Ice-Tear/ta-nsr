import trimesh
import argparse
import os
from pathlib import Path
import numpy as np
import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh
import argparse
from pathlib import Path

def clean_points_by_mask(points, scan):
    cameras = np.load('../data/DTU-NeuS/dtu_scan{}/cameras_sphere.npz'.format(scan))
    mask_lis = sorted(glob('../data/DTU-NeuS/dtu_scan{}/mask/*.png'.format(scan)))
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

def clean_mesh(old_mesh_path, scan, need_clean=True):
    old_mesh = trimesh.load(old_mesh_path)
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

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--DTU', type=str,  default='../data/Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    parser.add_argument('--need_clean', action='store_true')

    args = parser.parse_args()

    ply_file = args.input_mesh
	
    scan = os.path.basename(ply_file).split('.')[0][4:]
    
    Offical_DTU_Dataset = args.DTU
    out_dir = 'dtu_scan' + scan
    Path(out_dir).mkdir(parents=True, exist_ok=True)


    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")

    new_mesh = clean_mesh(ply_file, int(scan)) if args.need_clean else trimesh.load(ply_file)

    new_mesh.export(result_mesh_file)

    cmd = f"python eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    os.system(cmd)