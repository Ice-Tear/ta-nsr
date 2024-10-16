import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
from utils.misc import config_to_primitive
from models.utils.encoder import get_encoding
from models.utils.mlp import get_mlp
from .network_utils import update_module_step, scale_anything, get_activation



class SDFNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = get_encoding(3, config.xyz_encoding_config)
        self.n_output_dims = config.feature_dim
        self.mlp = get_mlp(self.encoder.n_output_dims, self.n_output_dims, config.mlp_network_config) 
        self.radius = config.radius
        self.gradient_config = config.gradient
        self.gradient_mode = config.gradient.mode
        self.fa_mode = config.gradient.get('fa_mode', 'none')
        if config.mlp_network_config.otype == 'FullyFusedMLP' or config.mlp_network_config.otype == 'CutlassMLP':
            self.sphere_bias = config.mlp_network_config.get('sphere_bias', 0.5)
        else:
            self.sphere_bias = 0.0

        ## finite epsilon ##
        self.start_step = self.config.xyz_encoding_config.start_step
        # compute the begin eps size
        self.base_res = np.floor(self.config.xyz_encoding_config.base_resolution * self.config.xyz_encoding_config.per_level_scale ** (self.config.xyz_encoding_config.start_level - 1))
        self.base_size = 2 * self.radius / self.base_res
        # compute the final eps size
        self.finest_res = np.floor(self.config.xyz_encoding_config.base_resolution * self.config.xyz_encoding_config.per_level_scale ** (self.config.xyz_encoding_config.n_levels - 1))
        self.finest_size = 2 * self.radius / self.finest_res
        self.final_size = self.finest_size / self.gradient_config.finest_size_ratio
        self.register_buffer('finite_difference_eps', torch.tensor(0.001, device='cuda:0'))
        if self.gradient_config.taps == 4:
            self.offsets = torch.as_tensor(
                    [[ 1.0, -1.0, -1.0],
                        [-1.0, -1.0,  1.0],
                        [-1.0,  1.0, -1.0],
                        [ 1.0,  1.0,  1.0]]
            , device='cuda:0')
        else:
            self.offsets = torch.as_tensor(
                        [[ 1.0,  0.0,  0.0],
                         [-1.0,  0.0,  0.0],
                         [ 0.0,  1.0,  0.0],
                         [ 0.0, -1.0,  0.0],
                         [ 0.0,  0.0,  1.0],
                         [ 0.0,  0.0, -1.0]]
                , device='cuda:0')

    def forward(self, points, with_feature=True, with_grad=True, with_laplace=False):
        ### compute gradient ###
        if self.gradient_mode == 'numerical':
            if self.gradient_config.taps == 4:

                eps = self.finite_difference_eps

                # print(self.offsets.shape)
                # print(points.shape)
                neighbors = (points[...,None,:] + self.offsets * eps).clamp(-self.radius, self.radius)
                neighbor_sdfs, neighbor_feats = self.forward_(neighbors, with_feature)
                grad = (self.offsets * neighbor_sdfs).sum(-2) / (4 * eps)
            
            elif self.gradient_config.taps == 6:
                eps = self.finite_difference_eps
                neighbors = (points[...,None,:] + self.offsets * eps).clamp(-self.radius, self.radius)
                neighbor_sdfs, neighbor_feats = self.forward_(neighbors, with_feature)
                grad = (self.offsets * neighbor_sdfs).sum(-2) / (2 * eps)
            else:
                raise NotImplementedError("Finite difference only support 4 or 6 taps.")
        
            ### Feature Aggregate ###
            normal = F.normalize(grad, p=2, dim=-1)
            if self.fa_mode == 'average':

                feats = neighbor_feats.sum(-2) * 0.25 if with_feature else None
                sdf = neighbor_sdfs.sum(-2).view(-1) * 0.25
                # sdf = feats[..., 0]
                # laplace = 2.0 * (neighbor_sdfs * 0.25 - sdf) / (eps ** 2)
                laplace = 2.0 * (neighbor_sdfs.sum(-1) * 0.25 - sdf) / (eps ** 2) if with_laplace else None
            elif self.fa_mode == 'tangent':

                with torch.no_grad():
                    new_offsets = torch.cross(normal.unsqueeze(1), self.offsets.unsqueeze(0), dim=-1)
                    weights = torch.norm(new_offsets,dim=-1) / torch.norm(new_offsets,dim=-1).sum(-1)[...,None]
                feats = (neighbor_feats * weights.unsqueeze(-1)).sum(-2) if with_feature else None
                sdf = (neighbor_sdfs * weights.unsqueeze(-1)).sum(-2).view(-1)
                # sdf = feats[..., 0]
                laplace = 2.0 * (neighbor_sdfs.sum(-2) * 0.25 - sdf) / (eps ** 2) if with_laplace else None
            elif self.fa_mode == 'none':

                sdf, feats = self.forward_(points, with_feature)
                # sdf = feats[..., 0]
                laplace = 2.0 * (neighbor_sdfs.sum(-2) * 0.25 - sdf) / (eps ** 2) if with_laplace else None
            else:
                raise NotImplementedError("This feature aggregate mode is not supported.")  
        
        elif self.gradient_mode == 'analytical':  # analytical
            with torch.inference_mode(torch.is_inference_mode_enabled() and not with_grad):
                with torch.set_grad_enabled(self.training or with_grad):
                    if with_grad:
                        if not self.training:
                            points = points.clone()
                        points.requires_grad_(True)
                    sdf, feats = self.forward_(points, with_feature)
                    
                    if with_grad:
                        grad = torch.autograd.grad(sdf, points, grad_outputs=torch.ones_like(sdf), 
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                        normal = F.normalize(grad, p=2, dim=-1)
                    if with_laplace:
                        laplace = torch.autograd.grad(grad, points, grad_outputs=torch.ones_like(grad), 
                                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                        laplace = laplace.sum(-1)
                    else:
                        laplace = None
        else:
            raise NotImplementedError("Only support analytical or numerical gradient.")
        
        out = {
            "sdf": sdf,
        }
        if with_feature:
            out.update({
                "features": feats,
            })  
        if with_grad:
            out.update({
                "grad": grad,
                "normal": normal,
            })
        if with_laplace:
            out.update({
                "laplace": laplace,
            }) 
        return out  # [sdf, features, sdf_grad, normal, sdf_laplace]
    
    def forward_(self, points, with_feature):
        points_contract = self.contract(points)  # Normalize to [0,1].
        enc_out = self.encoder(points_contract.view(-1, 3))
        points_enc = enc_out.view(*points.shape[:-1], enc_out.shape[-1])
        
        outs = self.mlp(points_enc, with_feature)
        sdfs, feats = outs
        sdfs = sdfs - self.sphere_bias
        if with_feature:
            return sdfs, feats - self.sphere_bias
        else:
            return sdfs, None
    
    def sdf(self, points):  # for marching cubes
        return self.forward(points, with_feature=False, with_grad=False, with_laplace=False)['sdf']

    def sdf_(self, points): # for grid prune
        return self.forward_(points, with_feature=False)[0].view(-1)

    def contract(self, points):
        return (points + self.radius) / (2 * self.radius)
    
    def update_step(self, current_step):
        # Compute epsilon
        if current_step< self.start_step:
            self.finite_difference_eps = torch.tensor(self.base_size)
        else:
            self.finite_difference_eps = torch.tensor(max(2 * self.radius / (self.base_res * np.power(self.config.xyz_encoding_config.per_level_scale, ((current_step - self.start_step) / self.config.xyz_encoding_config.update_steps))), self.final_size))

        # Update encoder
        update_module_step(self.encoder, current_step)

class RGBNetwork(nn.Module):
    def __init__(self, feats_dim, config):
        super().__init__()
        self.encoder = get_encoding(3, config.dir_encoding_config)
        self.include_xyz = config.mlp_network_config.get('include_xyz', True)
        self.include_normal = config.mlp_network_config.get('include_normal', True)
        # [pos, viewdirs_enc, features, normal]
        self.n_input_dims = int(self.include_xyz) * 3 + self.encoder.n_output_dims + feats_dim + int(self.include_normal) * 3
        self.mlp = get_mlp(self.n_input_dims, 3, config.mlp_network_config)
        self.activation = get_activation(config.get('color_activation',None))


    def forward(self, viewdirs, features, **args):
        # viewdirs = (viewdirs + 1.0) / 2.0
        viewdirs_enc = self.encoder(viewdirs)
        inputs = [viewdirs_enc, features]
        if self.include_xyz:
            inputs.append(args['points'])
        if self.include_normal:
            inputs.append(args['normal'])
        inputs = torch.cat(inputs, dim=-1)
        rgb = self.mlp(inputs).float()
        rgb = self.activation(rgb)
        return rgb
    
    def contract(self, points):
        pass

    def update_step(self, current_step):
        pass


class DensityNetWork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(
            n_input_dims=3, 
            n_output_dims=self.n_output_dims, 
            encoding_config=config.xyz_encoding_config,
            network_config=config.mlp_network_config
        )
        self.radius = config.radius
        self.density_activation = get_activation(config.get('density_activation', None))

    def forward(self, points):
        points_contract = self.contract(points)
        selector = ((points_contract > 0.0) & (points_contract < 1.0)).all(dim=-1)
        
        out = self.encoding_with_network(points_contract.view(-1, 3)).float()
        out = out.view(*points.shape[:-1], out.shape[-1])

        density, feature = out[..., 0], out
        density = self.density_activation(density) * selector
        return density, feature

    def contract(self, points):
        return scale_anything(points, (-self.radius, self.radius), (0, 1))

    def sdf(self, points):
        density, feature = self(points)
        return density

    def update_step(self, current_step):
        pass


class EncodingWithNetwork(nn.Module):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network
    
    def forward(self, x):
        return self.network(self.encoding(x))
    
    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)

def get_encoding_with_network(n_input_dims, n_output_dims, encoding_config, network_config):
    # input suppose to be range [0, 1]
    if encoding_config.otype in ['Fourier', 'ProgressiveBandHashGrid'] \
        or network_config.otype in ['VanillaMLP']:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        encoding_with_network = EncodingWithNetwork(encoding, network)
    else:
        with torch.cuda.device(0):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config)
            )
    return encoding_with_network
