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

@systems.register('TiNSR')
class NeuSSystem(BaseSystem):
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

        # rgb loss
        loss_rgb_l1 = F.l1_loss(out['pred_pixels'][out['rays_valid']], batch['rgb'][out['rays_valid']])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.rgb_weight

        # eikonal loss
        with torch.no_grad():
            if self.adaptive_weight > 0.0 and self.adaptive_weight < 1.0:
                rgb_mse = torch.mean((out['pred_pixels'] - batch['rgb']) ** 2, dim=1)
                diff_rays = self.adaptive_weight / (rgb_mse + self.adaptive_weight)
                # diff_rays = 1.0 - rgb_mse / rgb_mse.max()
                diff_pts = diff_rays[out['ray_indices']]  # out['ray_indices']是参与采样的点对于的光线
            else:
                diff_pts = 1.0
        # loss_eikonal = ((torch.linalg.norm(out['sdf_grad'], ord=2, dim=-1) - 1.)**2).mean()
        # loss_eikonal = eikonal_loss(out['sdf_grad'], diff_pts)
        loss_eikonal = eikonal_loss(out['sdf_grad'], out['relax_inside_sphere'], diff_pts)
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.eikonal_weight
        
        # mask loss
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        if self.dataset.apply_mask:
            loss_mask = binary_cross_entropy(opacity, batch['fg_mask'])
            self.log('train/loss_mask', loss_mask)
            loss += loss_mask * self.mask_weight

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.distortion_weight > 0.0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.distortion_weight

        if self.sparsity_weight > 0.0:
            loss_sparsity = torch.exp(-self.config.optimize.loss.sparsity_scale * out['random_sdfs'].abs()).mean()
            self.log('train/loss_sparsity', loss_sparsity)
            loss += self.sparsity_weight * loss_sparsity        

        self.log('loss', loss.item(), prog_bar=True)
        self.log('train/inv_s', out['inv_s'], prog_bar=True)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx):
        if self.val_images is None:
            self.val_images = self.dataset.real_image_idx
        real_image_idx = self.val_images[batch_idx]
        out, psnr_score = self.render_image(real_image_idx, self.dataset)
        self.save_image(image=out, image_idx=real_image_idx) 
        self.validation_step_outputs.append({
            'image': out,
            'psnr': psnr_score,
        })

    def on_validation_epoch_end(self):
        out = self.all_gather(self.validation_step_outputs)
        if self.trainer.is_global_zero:
            psnr_score = torch.mean(torch.stack([o['psnr'] for o in out]))
            self.log('val/psnr', psnr_score, prog_bar=True, rank_zero_only=True)
            self.val_psnr = 0.0
            self.export()

    def test_step(self, batch, batch_idx):
        real_image_idx = self.dataset.real_image_idx[batch_idx]
        out, psnr_score = self.render_image(real_image_idx, self.dataset)
        self.save_image(image=out, image_idx=real_image_idx)
        self.test_step_outputs.append({
            'image': out,
            'psnr': psnr_score,
        })

    def on_test_epoch_end(self):
        out = self.all_gather(self.test_step_outputs)
        if self.trainer.is_global_zero:
            psnr_score = torch.mean(torch.stack([o['psnr'] for o in out]))
            self.log('test/psnr', psnr_score, prog_bar=True, rank_zero_only=True)
            self.export()
            image_list = torch.stack([o['image'] for o in out]).detach().cpu().numpy()
            self.save_video(image_list, fps=30, save_format='gif', downfactor=2.)

    def predict_step(self, batch, batch_idx):
        pass

    def on_predict_end(self):
        print('Export Mesh...')
        self.export()

    @torch.no_grad()
    def render_image(self, image_idx, dataset):
        ''' 
            return: [gt_image, pred_image, depth_map, normal_map]
        '''
        # print(f'Rendering image {image_idx} ...')
        idx = dataset.real_image_idx.index(image_idx)  # get real image index
        dataloader = dataset.get_test_rays(image_idx=idx, chunk=self.ray_chunk)
        w, h = dataset.w, dataset.h
        gt_image = []
        pred_image = []
        depth_map = []
        normal_map = []
        if dataset.apply_mask:
            fg_masks = []
        for data in dataloader:
            gt_image.append(data['pixels'])
            if dataset.apply_mask:
                fg_masks.append((data['fg_mask']>0.999))
            out = self.model(data['rays'], data['color_bg'])
            
            if out is None:
                pred_image.append((data['color_bg'] * torch.ones_like(data['pixels'])))
                depth_map.append(torch.zeros((pred_image[-1].shape[0], 1), device='cuda:0')) 
                normal_map.append(torch.zeros_like(pred_image[-1]))
            else:
                pred_image.append(out['pred_pixels'])
                depth_map.append(out['depth'])
                normal_map.append(-out['pixels_normal'])
            del out

        gt_image = torch.cat(gt_image)
        pred_image = torch.cat(pred_image)
        depth_map = torch.cat(depth_map).cpu().numpy()
        normal_map = torch.cat(normal_map)

        # compute psnr
        # psnr_score = psnr(torch.cat(gt_image), torch.cat(pred_image), torch.cat(fg_masks) if dataset.apply_mask else None)
        psnr_score = psnr(gt_image, pred_image)
        # gt and pred image
        gt_image = pixels2img(gt_image, w, h)
        pred_image = pixels2img(pred_image, w, h)
        # depth
        depth_map = (np.concatenate(depth_map, axis=0).reshape(h, w, 1))
        depth_map = np.nan_to_num(depth_map)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255.).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        # normal
        normal_map = pixels2img(normal_map, w, h, data_range=(-1, 1))
        out = np.hstack([gt_image, pred_image, depth_map, normal_map])

        return out, psnr_score