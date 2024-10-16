import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import SequentialLR, ChainedScheduler

def scale_anything(x, input_scale, target_scale):
    if input_scale is None:
        input_scale = [x.min(), x.max()]
    x = (x  - input_scale[0]) / (input_scale[1] - input_scale[0])
    x = x * (target_scale[1] - target_scale[0]) + target_scale[0]
    return x

def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near.squeeze(-1), far.squeeze(-1)

def intersect_with_sphere(center, ray_unit, radius=1.0):
    ctc = (center * center).sum(dim=-1, keepdim=True)  # [...,1]
    ctv = (center * ray_unit).sum(dim=-1, keepdim=True)  # [...,1]
    b2_minus_4ac = ctv ** 2 - (ctc - radius ** 2)
    dist_near = -ctv - b2_minus_4ac.sqrt()
    dist_far = -ctv + b2_minus_4ac.sqrt()
    dist_near.relu_()
    outside = dist_near.isnan()
    outside = torch.where(dist_near.isnan())[0]
    dist_near[outside], dist_far[outside] = 1.0, 1.2
    return dist_near.squeeze(-1), dist_far.squeeze(-1), outside.long()

def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([[0]]).to(ray_indices)
        t_end = torch.Tensor([[0]]).to(ray_indices)
    return ray_indices, t_start, t_end


def update_module_step(m, current_step):
    if hasattr(m, 'update_step'):
        m.update_step(current_step)

class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)

def parse_scheduler(config, optimizer):
    # interval = config.get('interval', 'epoch')
    # assert interval in ['epoch', 'step']
    interval = 'step'
    if config.name == 'SequentialLR':
        scheduler = {
            'scheduler': SequentialLR(optimizer, [parse_scheduler(conf, optimizer)['scheduler'] for conf in config.schedulers], milestones=config.milestones),
            'interval': interval
        }
    elif config.name == 'Chained':
        scheduler = {
            'scheduler': ChainedScheduler([parse_scheduler(conf, optimizer)['scheduler'] for conf in config.schedulers]),
            'interval': interval
        }
    else:
        scheduler = {
            'scheduler': getattr(lr_scheduler, config.name)(optimizer, **config.args),
            'interval': interval
        }
    return scheduler

def parse_optimizer(config, params_dict):
    params_list = []
    if hasattr(config, 'params'):
        for key, value in params_dict.items():
            params_list.append({'params': value, 'name': key, **config.params[key]})
    else:
        params_list = params_dict
    if config.name in ['FusedAdam']:
        # If you have a good cpu, apex is faster than torch.
        import apex
        optim = getattr(apex.optimizers, config.name)(params_list, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params_list, **config.args)
    return optim