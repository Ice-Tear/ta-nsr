import math
import torch
import torch.nn.functional as F
from models.utils.modules import SDFNetwork, RGBNetwork, DensityNetWork
from models.utils.network_utils import update_module_step, near_far_from_sphere, validate_empty_rays

from nerfacc import contract_inv, ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
import models

@models.register('NeRF')
class NeRF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.geometry_network = DensityNetWork(config.geometry)
        self.color_network = RGBNetwork(self.geometry_network.n_output_dims, config.color)

        if config.background.enabled:
            pass
            # self.background_network = 
        
        # OccGird
        self.init_occgrid()

        # parameters dict
        self.params_dict = {
            "geometry": self.geometry_network.parameters(),
            "color": self.color_network.parameters(),
            }

        
    def forward(self, rays, color_bg=None):
        rays_o, rays_d = rays
        n_rays = len(rays_o)

        ### Occgrid Ray Sample ###
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_network(positions)
            return density[...,None]
        
        ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.estimator,
                sigma_fn=sigma_fn,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.training,
                cone_angle=0.0,
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        points = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        # if points.shape[0] == 0: # FullyFusedMLP doesn't support 0 inputs
        #     return None

        ### forward ###
        density, feature = self.geometry_network(points)
        rgbs = self.color_network(t_dirs, feature)  # [pos, viewdirs, features, normal]
        
        ### rendering ###
        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        pixels_color = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)

        if color_bg is not None:
            pixels_color = pixels_color + color_bg * (1.0 - opacity)
        
        out = {
            "pred_pixels": pixels_color,
            "opacity": opacity,
            "depth": depth,
            "weights": weights.view(-1),
            "points": midpoints.view(-1),
            "intervals": intervals.view(-1),
            "ray_indices": ray_indices.view(-1),
            "rays_valid": opacity>0.0,
            "n_samples": len(t_starts),
        }
        return out

    def init_occgrid(self, cameras=None):
        config = self.config.render.sampler
        self.register_buffer('scene_aabb', 
                             torch.as_tensor([-config.scene_aabb, -config.scene_aabb, -config.scene_aabb, 
                                               config.scene_aabb,  config.scene_aabb,  config.scene_aabb], dtype=torch.float32))
        self.estimator = OccupancyGrid(
            roi_aabb=self.scene_aabb.contiguous(), 
            resolution=config.resolution, 
            contraction_type=ContractionType.AABB)
        
        self.render_step_size = config.render_step_size
        # self.render_step_size = 1.732 * 2 * config.scene_aabb / 512
        self.grid_prune = config.grid_prune
        self.grid_prune_occ_thre = config.grid_prune_occ_thre
        # mark_invisible_cells
        if cameras is not None:
            K, pose, width, height = cameras

            self.estimator.mark_invisible_cells(K, pose, width, height)
            print((self.estimator.occs == -1).sum())
            print((self.estimator.occs == 0).sum())
            assert False

    def random_sample_from_occgrid(self, num_points=1024):
        indices = self.estimator._sample_uniform_and_occupied_cells(num_points//2)
        # infer occupancy: density * step_size
        grid_coords = self.estimator.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.estimator.resolution
        if self.estimator._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self.estimator._roi_aabb,
            type=self.estimator._contraction_type,
        )
        return x

    def update_step(self, current_step):
        # Compute epsilon and update HashEncoder
        update_module_step(self.geometry_network, current_step)  

        # Update OccGrid
        def occ_eval_fn(x):
            density, _ = self.geometry_network(x)
            return density[..., None] * self.render_step_size
        
        if self.training and self.grid_prune:
            self.estimator.every_n_step(
                step=current_step,
                occ_eval_fn=occ_eval_fn,
            )