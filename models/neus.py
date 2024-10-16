import math
import torch
import torch.nn.functional as F
from models.utils.modules import SDFNetwork, RGBNetwork, DensityNetWork
from models.utils.network_utils import update_module_step, near_far_from_sphere, validate_empty_rays

from nerfacc import contract_inv, ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
import models

@models.register('TiNSR')
class TiNSR(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.geometry_network = SDFNetwork(config.geometry)
        self.color_network = RGBNetwork(self.geometry_network.n_output_dims, config.color)
        self.s_var = torch.nn.Parameter(torch.tensor(config.variance.init_val * 10.0, dtype=torch.float32))
        if self.config.background.enabled:
            self.bg_geometry_network = DensityNetWork(config.bg_geometry)
            self.bg_color_network = RGBNetwork(config.bg_color)
        
        self.with_laplace = False
        # OccGird
        self.init_occgrid()
        self.register_buffer('cos_anneal_ratio', torch.tensor(1.0).to(self.scene_aabb))

        # parameters dict
        self.params_dict = {
            "geometry": self.geometry_network.parameters(),
            "color": self.color_network.parameters(),
            "variance": self.s_var,
            }

        
    def forward(self, rays, color_bg=None):
        rays_o, rays_d = rays
        n_rays = len(rays_o)
        
        ### Occgrid Ray Sample ###
        def alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            t_intervals = t_ends - t_starts
            points = t_origins + t_dirs * midpoints
            with torch.no_grad():
                out = self.geometry_network(points, with_feature=False, with_grad=True, with_laplace=False)
                sdf, sdf_grad = out['sdf'], out['grad']
                alpha = self.get_alpha(sdf, sdf_grad, t_dirs, t_intervals)[...,None]
            return alpha
        
        with torch.no_grad():
            t_mins, t_maxs = near_far_from_sphere(rays_o, rays_d)
            ray_indices, t_starts, t_ends = ray_marching(
                    rays_o, rays_d,
                    t_min=t_mins, t_max=t_maxs,
                    scene_aabb=self.scene_aabb,
                    grid=self.estimator if self.grid_prune else None,
                    alpha_fn=alpha_fn,
                    # alpha_fn=None,
                    # near_plane=None, far_plane=None,
                    render_step_size=self.render_step_size,
                    stratified=self.training,
                    cone_angle=0.0,
                    alpha_thre=0.01
                )
            
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        points = t_origins + t_dirs * midpoints
        t_intervals = t_ends - t_starts
        pts_norm = torch.linalg.norm(points, ord=2, dim=-1, keepdim=True)
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        if points.shape[0] == 0: # FullyFusedMLP doesn't support 0 inputs
            return None

        ### forward ###
        out = self.geometry_network(points, with_feature=True, with_grad=True, with_laplace=self.with_laplace)
        sdf, feats, grad, normal = out['sdf'], out['features'], out['grad'], out['normal']
        rgbs = self.color_network(t_dirs, feats, points=points, normal=normal)  # [pos, viewdirs, features, normal]
        
        ### rendering ###
        # object rendering
        alphas = self.get_alpha(sdf, grad, t_dirs, t_intervals)[...,None]
        weights = render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        pixels_color = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)

        # background rendering
        
        if color_bg is not None:
            pixels_color = pixels_color + color_bg * (1.0 - opacity)
        pixels_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        pixels_normal = F.normalize(pixels_normal, p=2, dim=-1)

        # maybe can reduce floaters
        random_points = self.random_sample_from_occgrid(num_points=1024)
        random_sdfs = self.geometry_network.sdf_(random_points)
        

        out = {
            "pred_pixels": pixels_color,
            "pixels_normal": pixels_normal,
            "opacity": opacity,
            "depth": depth,
            "weights": weights.view(-1),
            "points": midpoints.view(-1),
            "intervals": t_intervals.view(-1),
            "ray_indices": ray_indices.view(-1),
            "rays_valid": torch.where(opacity>0.0)[0],
            "sdf": sdf,
            "sdf_grad": grad,
            "relax_inside_sphere": relax_inside_sphere.view(-1),
            "sdf_laplace": out['laplace'] if self.with_laplace else None,
            "n_samples": len(t_starts),
            "inv_s": self.s_var.exp().item(),
            "random_sdfs": random_sdfs,
        }
        # Lipshitz Loss
        if self.config.color.mlp_network_config.otype == 'LipshitzMLP':
            lipshitz = self.color_network.mlp.layers.lipshitz_bound_full()
            out.update({
                'lipshitz': lipshitz
            })
        return out

    def render_background(self):
        return

    def get_alpha(self, sdf, grad, dirs, dists):
        inv_s = self.s_var.exp().clamp(1e-6, 1e6)
        dSDF_dt = (grad * dirs).sum(-1, keepdim=True)
        
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        dSDF_dt = -(F.relu(-dSDF_dt * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                    F.relu(-dSDF_dt) * self.cos_anneal_ratio)  # always non-positive    

        cdf = torch.sigmoid(inv_s * sdf[...,None])
        rho = torch.clamp((inv_s * (cdf - 1.0) * dSDF_dt), 0.0)
        alpha = (1 - torch.exp(-rho * dists)).view(-1)  # dists dim is [n_samples,] in the latest version of Nerfacc.
        return alpha
            
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
        # self.render_step_size = 1.732 * 2 * config.scene_aabb / self.config.render.num_samples_per_ray
        
        if self.config.background.enabled:
            self.bg_render_step_size = 1e-2
            self.bg_near_plane, self.bg_far_plane = 0.1, 1e3
            self.bg_cone_angle = 10**(math.log10(self.bg_far_plane) / self.config.render.num_samples_per_ray_bg) - 1.
            self.bg_estimator = OccupancyGrid(
                roi_aabb=self.scene_aabb.contiguous(), 
                resolution=config.resolution * 2, 
                contraction_type=ContractionType.UN_BOUNDED_SPHERE)

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

        # Update cos_anneal_ratio
        cos_anneal_end = self.config.render.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = torch.tensor(1.0) if cos_anneal_end == 0 else torch.tensor(min(1.0, current_step / cos_anneal_end)).to(self.scene_aabb)

        # Update OccGrid
        def occ_eval_fn(x):
            sdf = self.geometry_network.sdf_(x)
            inv_s = self.s_var.exp().clamp(1e-6, 1e6)
            rho = torch.clamp((inv_s * (1.0 - torch.sigmoid(inv_s * sdf[...,None]))), 0.0)
            alpha = (1 - torch.exp(-rho * self.render_step_size))
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.bg_geometry_network(x)
            return density[...,None] * self.bg_render_step_size
         
        if self.training and self.grid_prune:
            self.estimator.every_n_step(
                step=current_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.grid_prune_occ_thre,
                n=8,
                )
            if self.config.background.enabled:
                self.bg_estimator.every_n_step(
                    step=current_step,
                    occ_eval_fn=occ_eval_fn_bg,
                    occ_thre=self.grid_prune_occ_thre,
                    # n=1,
                    )