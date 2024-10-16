import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn
import tet_utils
from utils.misc import config_to_primitive
from .network_utils import update_module_step
    
def get_encoding(n_input_dims, config):
    if config.otype == 'ProgressiveHashGrid':
        encoding = ProgressiveHashGrid(n_input_dims, config_to_primitive(config))
    elif config.otype == 'SphericalHarmonics':
        encoding = SHEncoder(n_input_dims, config_to_primitive(config))
    elif config.otype == 'HashGrid' or config.otype == 'Frequency':
        with torch.cuda.device(0):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config))
    else:
        assert False, 'We do not support this way of encoding.'

    encoding = CompositeEncoding(encoding, include_xyz=config.get('include_xyz', False), xyz_scale=2., xyz_offset=-1.)
    return encoding    

class SHEncoder(nn.Module):
    def __init__(self, in_channels, config):
        ''' Unlike the tcnn's encoder, this encoder doesn't need dirs in [0, 1].
        '''
        super().__init__()
        assert in_channels == 3
        self.degree = config['degree']
        self.n_output_dims = self.degree  ** 2
        self.n_input_dims = in_channels

    def forward(self, dirs):
        # tet_utils is little faster than torch version.
        out = tet_utils.spherical_harmonics(dirs, self.degree, 0)  # 0 for float16, 1 for float32
        return out

class ProgressiveHashGrid(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config['otype'] = 'HashGrid'
        with torch.cuda.device(0):
            self.encoding = tcnn.Encoding(in_channels, encoding_config)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_levels = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']
        self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        self.current_level = self.start_level
        self.register_buffer('mask', torch.zeros(self.n_levels * self.n_features_per_level, dtype=torch.float32))
        # self.mask = torch.zeros(self.n_level * self.n_features_per_level, dtype=torch.float32)

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, current_step):
        current_level = min(self.start_level + max(current_step - self.start_step, 0) // self.update_steps, self.n_levels)
        if current_level > self.current_level:
            # print(f'Update HashGrid level to {current_level}')
            pass
        self.current_level = current_level
        self.mask[:self.current_level * self.n_features_per_level] = 1.

class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super().__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
    
    def forward(self, x):
        return self.encoding(x) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x)], dim=-1)

    def update_step(self, current_step):
        update_module_step(self.encoding, current_step)