import os
from shutil import copy, copytree
from utils.misc import dump_config

from lightning.pytorch.callbacks import TQDMProgressBar, Callback

class CustomProgressBar(TQDMProgressBar):
    def __init__(self, total_iters ,refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        self.total_iters = total_iters

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.total_iters
        return bar
    
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items

class CodeSnapShotCallback(Callback):
    def __init__(self, config, need_save=True):
        self.config = config
        self.need_save = need_save

    def file_backup(self):
        code_dir = os.path.join(self.result_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)

        # save origin code
        save_list = self.config.general.save_list
        for file_name in save_list:     
            if os.path.isdir(file_name):
                copytree(file_name, os.path.join(code_dir, file_name))
            else:
                copy(file_name, code_dir)

        # save resolved config
        dump_config(os.path.join(self.result_dir, 'config.yaml'), self.config)

    def on_fit_start(self, trainer, pl_module):
        if self.need_save:
            self.result_dir = pl_module.result_dir
            self.file_backup()
