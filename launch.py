import os
import argparse
from datetime import datetime
from utils.misc import load_config, seed_everything

# import cumcubes
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--case', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='If offered, the runner will resume from this ckpt.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    n_gpus = len(args.gpu.split(','))

    import torch
    import datasets
    import systems
    from lightning.pytorch import Trainer
    from systems.callbacks import CustomProgressBar, CodeSnapShotCallback
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    import sys

    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        strategy = 'ddp_find_unused_parameters_false'


    if args.case is not None:
        extras.append(f'dataset.case_name={args.case}')
    config = load_config(args.config, cli_args=extras)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    result_dir = os.path.dirname(args.ckpt) if args.ckpt is not None else os.path.join(config.general.result_dir, datetime.now().strftime('@%Y%m%d-%H%M%S'))
    
    dm = datasets.make('dtu', config.dataset)
    system = systems.make(config.general.system, config, load_from_checkpoint=args.ckpt)

    system.result_dir = result_dir
    
    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=result_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapShotCallback(config, need_save=args.ckpt is None),
            CustomProgressBar(total_iters=config.trainer.max_steps, refresh_rate=1),
            ]
    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(result_dir, name='logs', version=''),
        ]

    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        # strategy=strategy,
        **config.trainer
    )
    # python launch.py --config exp/TiNSR-dtu_scan24-wmask/@20240227-112755/config.yaml --test --ckpt exp/TiNSR-dtu_scan24-wmask/@20240227-112755/epoch=0-step=10000.ckpt
    if args.train:
        trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.ckpt)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.ckpt)
    elif args.export:
        trainer.predict(system, datamodule=dm, ckpt_path=args.ckpt)
