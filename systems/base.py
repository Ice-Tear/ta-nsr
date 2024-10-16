import os
import cv2
import imageio
import lightning as pl
import models
from utils.mesh import MarchingCubeHelper
from utils.misc import get_rank
from models.utils.network_utils import parse_optimizer, parse_scheduler, update_module_step

class BaseSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = models.make(self.config.model.name, self.config.model)

    def prepare(self):
        pass

    def preprocess_data(self, batch, stage):
        pass

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')
        update_module_step(self.model, self.global_step)

    def on_validation_batch_start(self, batch, batch_idx):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')
        # update_module_step(self.model, self.global_step)
    
    def on_test_batch_start(self, batch, batch_idx):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')
        # update_module_step(self.model, self.global_step)

    def on_predict_batch_start(self, batch, batch_idx):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')
        # update_module_step(self.model, self.global_step)
    
    def configure_optimizers(self):
        optimizer = parse_optimizer(self.config.optimize.optimizer, params_dict=self.model.params_dict)
        scheduler = parse_scheduler(self.config.optimize.scheduler, optimizer)
        return [optimizer], [scheduler]

    def export(self):
        self.mesh_helper = MarchingCubeHelper(self.model, self.config.model.render.isosurface, scale_and_center=self.dataset.scale_and_center)
        save_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'it-{self.global_step}-mc{self.mesh_helper.resolution}.ply')
        self.mesh_helper.save_mesh(filename)

    def save_image(self, image, image_idx):
        save_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)            
        filename = os.path.join(save_dir, f'{image_idx}-it-{self.global_step}.png')
        cv2.imwrite(filename, image)  
    
    def save_video(self, image_list, fps=30, save_format='mp4', downfactor=1.):
        assert save_format in ['mp4', 'gif']
        image_list = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), (int(i.shape[1] / downfactor), int(i.shape[0] / downfactor))) for i in image_list]
        save_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)   
        filename = os.path.join(save_dir, f"it{self.global_step}-test.{save_format}")   
        if save_format == 'mp4':
            imageio.mimsave(filename, image_list, fps=fps)
        elif save_format == 'gif':
            imageio.mimsave(filename, image_list, duration=(1000 / fps), loop=0)