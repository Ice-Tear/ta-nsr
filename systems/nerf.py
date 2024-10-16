import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch_efficient_distloss import flatten_eff_distloss
from utils.misc import load_config, dump_config, seed_everything, pixels2img
from systems.losses import rgb_loss, eikonal_loss, binary_cross_entropy, curvature_loss, psnr
from systems.base import BaseSystem
import systems

@systems.register('nerf')
class NeRFSystem(BaseSystem):
    def prepare(self):
        self.val_images = self.config.dataset.get('val_images', None)
        self.ray_chunk = self.config.model.render.get('ray_chunk', 2048)

        # Sampler parameters
        self.train_num_rays = self.config.model.render.train_num_rays
        self.num_samples_per_ray = self.config.model.render.num_samples_per_ray
        self.total_num_samples = self.train_num_rays * self.num_samples_per_ray
        self.max_train_num_rays = self.config.model.render.max_train_num_rays

        # Loss Weights
        self.rgb_weight = self.config.optimize.loss.rgb_weight
        self.mask_weight = self.config.optimize.loss.get('mask_weight', 0.0)
        self.eikonal_weight = self.config.optimize.loss.get('eikonal_weight', 0.0)
        self.adaptive_weight = self.config.optimize.loss.get('adaptive_weight', 0.0)
        self.curvature_weight = self.config.optimize.loss.get('curvture_weight', 0.0)
        self.opaque_weight = self.config.optimize.loss.get('opaque_weight', 0.0)
        self.distortion_weight = self.config.optimize.loss.get('distortion_weight', 0.0)
        self.lipshitz_weight = self.config.optimize.loss.get('lipshitz_weight', 0.0)
        self.sparsity_weight = self.config.optimize.loss.get('sparsity_weight', 0.0)

        self.val_psnr = 0.0
        self.test_psnr = 0.0

    def forward(self, batch):
        return self.model(batch['rays'], batch['color_bg'])
    
    def preprocess_data(self, batch, stage):
        if stage in ['train']:
            data = self.dataset.get_train_rays(self.train_num_rays)

            rays = data['rays']
            rgb = data['pixels']
            fg_mask = data['fg_mask']

            if self.dataset.apply_mask:
                rgb = rgb * fg_mask[...,None] + data['color_bg'] * (1 - fg_mask[...,None])
            
            batch.update({
                'rays': rays,
                'rgb': rgb,
                'fg_mask': fg_mask,
                'color_bg': data['color_bg']
            })    

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        if self.config.model.render.dynamic_ray_sampling:
            self.train_num_rays = int(self.train_num_rays * (self.total_num_samples / float(out['n_samples'])))
            self.train_num_rays = min(self.train_num_rays, self.max_train_num_rays)
        # min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        # rgb loss
        loss_rgb = F.mse_loss(out['pred_pixels'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.rgb_weight

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.distortion_weight > 0.0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.distortion_weight 

        self.log('loss', loss.item(), prog_bar=True)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx):
        if self.val_images is None:
            self.val_images = self.dataset.real_image_idx
        real_image_idx = self.val_images[batch['index'].item()]
        out, psnr_score = self.render_image(real_image_idx, self.dataset)
        save_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)            
        filename = os.path.join(save_dir, f'{real_image_idx}-it-{self.global_step}.png')
        cv2.imwrite(filename, out)   
        self.val_psnr += psnr_score
        # pass

    def on_validation_epoch_end(self):
        # self.export()
        self.log('val/psnr', self.val_psnr / self.trainer.limit_val_batches, prog_bar=True, rank_zero_only=True)
        self.val_psnr = 0.0
    
    def test_step(self, batch, batch_idx):
        real_image_idx = self.dataset.real_image_idx[batch['index']]
        out, psnr_score = self.render_image(real_image_idx, self.dataset)
        save_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)            
        filename = os.path.join(save_dir, f'{real_image_idx}-it-{self.global_step}.png')
        cv2.imwrite(filename, out)   
        self.test_psnr += psnr_score
        # pass

    def on_test_epoch_end(self):
        self.log('test/psnr', self.test_psnr / self.dataset.n_images, prog_bar=True, rank_zero_only=True)
        self.export()

    def predict_step(self, batch, batch_idx):
        pass

    def on_predict_end(self):
        print('Export Mesh...')
        self.export()

    @torch.no_grad()
    def render_image(self, image_idx, dataset):
        ''' 
            return: [gt_image, pred_image, depth_map]
        '''
        idx = dataset.real_image_idx.index(image_idx)  # get real image index
        dataloader = dataset.get_test_rays(image_idx=idx, chunk=self.ray_chunk)
        w, h = dataset.w, dataset.h
        gt_image = []
        pred_image = []
        depth_map = []
        fg_masks = []
        for data in dataloader:
            gt_image.append(data['pixels'].detach().cpu())
            fg_masks.append((data['fg_mask']>0.999).detach().cpu())
            out = self.model(data['rays'], data['color_bg'])
            if out is None:
                    pred_image.append((data['color_bg'] * torch.ones_like(data['pixels'])).detach().cpu())
                    depth_map.append(torch.zeros((pred_image[-1].shape[0], 1)).cpu().numpy()) 
            else:
                pred_image.append(out['pred_pixels'].detach().cpu())
                depth_map.append(out['depth'].detach().cpu().numpy())
            del out

        # compute psnr
        psnr_score = psnr(torch.cat(gt_image), torch.cat(pred_image), torch.cat(fg_masks))
        # gt and pred image
        gt_image = pixels2img(gt_image, w, h)
        pred_image = pixels2img(pred_image, w, h)
        # depth
        depth_map = (np.concatenate(depth_map, axis=0).reshape(h, w, 1))
        depth_map = np.nan_to_num(depth_map)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255.).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        out = np.hstack([gt_image, pred_image, depth_map])
        return out, psnr_score