import os
from omegaconf import OmegaConf
import torch
import cv2
import random
import numpy as np
import gc
import tinycudann as tcnn

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n), replace=True)
OmegaConf.register_new_resolver('add', lambda a, b: a + b, replace=True)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b, replace=True)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver('div', lambda a, b: a / b, replace=True)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver('len', lambda a: len(a), replace=True)
OmegaConf.register_new_resolver('get', lambda a, idx: a[idx], replace=True)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p), replace=True)
# ======================================================= #


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    if conf.get('parent_config') is not None:
        parent_config = OmegaConf.load(conf.get('parent_config'))
        conf = OmegaConf.merge(parent_config, conf)
    OmegaConf.resolve(conf)
    return conf

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)

def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)
        
def pixels2img(pixels, w, h, data_range=(0, 1)):
    image = (pixels.reshape(h, w, 3).cpu().numpy()).clip(data_range[0], data_range[1])
    image = (image - data_range[0]) / (data_range[1] - data_range[0])
    image = (image * 255.).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0
