import torch
import numpy as np
import torch.nn.functional as F
import collections

Rays = collections.namedtuple("Rays", ("rays_o", "rays_d"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    u, v = torch.meshgrid(
        torch.arange(W) + pixel_center,
        torch.arange(H) + pixel_center,
        indexing='xy'
    )

    directions = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(v)], -1) # (H, W, 3)

    return directions

def get_ray_directions_test(W, H, fx, fy, cx, cy, intrinsic, use_pixel_centers=True):

    y_range = torch.arange(H, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(W, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    
    directions = img2cam(to_hom(xy_grid), intrinsic[:3,:3])
    # print(directions)
    # print(directions.shape)
    # assert False
    # directions = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(v)], -1) # (H, W, 3)
    return directions

def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
        
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    rays_d = F.normalize(rays_d, p=2, dim=-1)  # normalize viewdirs 
    rays = Rays(rays_o=rays_o, rays_d=rays_d)
    return rays
