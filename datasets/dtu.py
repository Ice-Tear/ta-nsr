import os
import json
import math
import numpy as np
from PIL import Image
import cv2
import threading
import queue
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
import collections

Rays = collections.namedtuple("Rays", ("rays_o", "rays_d"))

import lightning as pl

import datasets
from .utils import get_ray_directions, get_rays
from utils.misc import get_rank

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
    up = rot_axis
    rot_dir = torch.cross(rot_axis, cam_center)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class DTUDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.device = 'cuda:0'
        self.num_workers = 16
        self.background_color = 'random'

        self.root_dir = os.path.join(config.root_dir, config.case_name)
        transforms_path = os.path.join(self.root_dir, f'transform_{self.split}.json')
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
            
        self.transforms = transforms
        w, h = transforms['w'], transforms['h']

        if config.get('img_downscale', False):
            w, h = int(w / config.img_downscale + 0.5), int(h / config.img_downscale + 0.5)

        self.factor = w / transforms['w']
        self.w = w
        self.h = h
        self.img_wh = (w, h)

        # use for transform mesh to the real world
        self.scale_and_center = [transforms['sphere_radius'], torch.Tensor(transforms['sphere_center']).to(self.rank)]

        self.has_mask = True
        self.apply_mask = self.config.apply_mask
        

        self.images_list = [os.path.join(self.root_dir, frame['file_path']) for frame in transforms['frames']]
        self.real_image_idx = [int(os.path.basename(file_name).split('.')[0]) for file_name in self.images_list]  # 原文件的索引

        self.n_images = len(self.images_list)

        self.all_images = self.preload_with_multi_threads(self.load_image, num_workers=self.num_workers, data_str='images')
        if self.apply_mask:
            self.all_fg_masks = self.all_images[...,3]
        self.all_images = self.all_images[...,:3]

        self.directions = self.preload_with_multi_threads(self.load_intrinsic, num_workers=self.num_workers, data_str='intrinsics')
        self.all_c2w = self.preload_with_multi_threads(self.load_camera, num_workers=self.num_workers, data_str='cameras matrix')
        print(f'Load {self.split} data: End')


    def get_all_cameras(self):
        all_intrinsics = self.preload_with_multi_threads(self.load_intr, num_workers=self.num_workers, data_str='all_intrinsic')
        return all_intrinsics, self.all_c2w, self.w, self.h

    def load_image(self, idx):
        rgba = cv2.imread(self.images_list[idx], cv2.IMREAD_UNCHANGED)
        if self.factor != 1.0 :
            rgba = cv2.resize(rgba, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        rgba = torch.from_numpy(rgba.astype(np.float32)) / 255.
        return rgba

    def load_camera(self, idx):
        c2w = torch.from_numpy(np.array(self.transforms['frames'][idx]['transform_matrix'])).float()
        return c2w[:3,:4]

    def load_intr(self, idx):
        intr = torch.from_numpy(np.array(self.transforms['frames'][idx]['intrinsic_matrix'])).float()
        return intr[:3,:3]
        
    def load_intrinsic(self, idx):
        # Get Intrinsic
        fx = self.transforms['frames'][idx]['intrinsic_matrix'][0][0] * self.factor
        fy = self.transforms['frames'][idx]['intrinsic_matrix'][1][1] * self.factor
        cx = self.transforms['frames'][idx]['intrinsic_matrix'][0][2] * self.factor
        cy = self.transforms['frames'][idx]['intrinsic_matrix'][1][2] * self.factor
        directions = get_ray_directions(self.w, self.h, fx, fy, cx, cy)        
        return directions

    def _preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        # Keep preloading data in parallel.
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_with_multi_threads(self, load_func, num_workers, data_str='images'):
        data_list = [None] * self.n_images

        q = queue.Queue(maxsize=self.n_images)
        idx_tqdm = tqdm.tqdm(range(self.n_images), desc=f"Loading {data_str}", leave=False)
        for i in range(self.n_images):
            q.put(i)
        lock = threading.Lock()
        for ti in range(num_workers):
            t = threading.Thread(target=self._preload_worker,
                                 args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert all(map(lambda x: x is not None, data_list))
        
        
        data_list = torch.stack(data_list, dim=0)
        data_list = data_list.float().to(self.rank)
        return data_list           
    
    def preprocess(self, data):
        # background color
        if self.background_color == "random":
            color_bg = torch.rand(3, device=self.device)
        elif self.background_color == "white":
            color_bg = torch.ones(3, device=self.device)
        elif self.background_color == "black":
            color_bg = torch.zeros(3, device=self.device)
        
        data.update({
            "color_bg": color_bg,
        })
        return data

    @torch.no_grad()
    def get_train_rays(self, num_rays):
        '''
        Generate random rays at world space from random cameras.
        '''
        assert self.split == 'train', f'This is the {self.split} dataset! Only the train dataset has this function. If you want to render the whole image, please use get_test_rays().'
        
        image_idx, u, v = self.uv_generator(num_rays=num_rays, unique=True)

        data = self.get_rays_from_images(image_idx, u, v)
        data = self.preprocess(data)
        return data

    @torch.no_grad()
    def get_test_rays(self, image_idx, chunk=None):
        '''
        Generate all rays at world space from one camera.
        '''
        u, v = torch.meshgrid(
            torch.arange(self.w, device=self.device),
            torch.arange(self.h, device=self.device),
            indexing="xy",
        )
        u = u.flatten()
        v = v.flatten()      
        if chunk is None:
            chunk = len(u)

        u = u.split(chunk)
        v = v.split(chunk)
        block_idx = 0
        while block_idx < len(u):
            data = self.get_rays_from_images([image_idx], u[block_idx], v[block_idx])
            # just use white during inference
            color_bg = torch.ones(3, device=self.device)  
            if self.apply_mask:
                data['pixels'] = data['pixels'] * data['fg_mask'][...,None] + color_bg * (1 - data['fg_mask'][...,None])
            data.update({
                "color_bg": color_bg,
            })
            yield data
            block_idx += 1

    def get_rays_from_images(self, image_idx, u, v):

        c2w = self.all_c2w[image_idx]
        directions = self.directions[image_idx, v, u]

        rays = get_rays(directions, c2w)      # [num_rays, 3]
        rgb = self.all_images[image_idx, v, u]          # [num_rays, 3]
        fg_mask = self.all_fg_masks[image_idx, v, u] if self.apply_mask else None   # [num_rays]

        return {
            "rays": rays,  # (rays_o, rays_d)
            "pixels": rgb,
            "fg_mask": fg_mask,
        }
    
    def get_rays_from_new_view(self, cameras_path):
        pass

    def uv_generator(self, num_rays, unique=True):
        image_idx = torch.randint(
            0, 
            self.n_images, 
            size=(num_rays,), 
            device=self.device
            )
        u = torch.randint(
                0, self.w, size=(num_rays,), device=self.device
            )
        v = torch.randint(
                0, self.h, size=(num_rays,), device=self.device
            )
        if unique:
            unique_uv = torch.stack([image_idx, u, v], dim=-1)
            unique_uv = torch.unique(unique_uv, dim=0)
            return unique_uv[...,0], unique_uv[...,1], unique_uv[...,2]
        else:
            return image_idx, u, v
        
class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return self.n_images
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('dtu')
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DTUIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DTUDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test', 'predict']:
            self.test_dataset = DTUDataset(self.config, self.config.get('test_split', 'test'))

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        # sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            # pin_memory=True,
            # sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=None)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=len(self.test_dataset))       