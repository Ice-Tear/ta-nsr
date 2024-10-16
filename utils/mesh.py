import numpy as np
import torch
import mcubes
import cumcubes
from utils.misc import cleanup

################################ CUDA Version ################################
class MarchingCubeHelper:
    def __init__(self, model, config, scale_and_center=None):
        ### Marching Cubes Config ###
        self.resolution = config.get('resolution', 512)
        self.threshold = config.get('threshold', 0.0)
        self.block_size = config.get('block_size', 64)
        self.export_color = config.get('export_color', False)
        ### 
        self.device = 'cuda:0'
        self.use_cuda = config.get('cuda', True)
        self.model = model
        self.sdf_func = model.geometry_network.sdf
        self.color_func = self.extract_color if self.export_color else None
        self.bound_min = model.scene_aabb[:3].to(self.device)
        self.bound_max = model.scene_aabb[3:].to(self.device)
        if scale_and_center is not None:
            self.scale, self.center = scale_and_center
            self.center = self.center.cpu() if not self.use_cuda else self.center

        # todo: export texture map and mtl
       
    def extract_fields(self):
        N = self.block_size
        resolution = self.resolution
        X = torch.linspace(self.bound_min[0], self.bound_max[0], resolution, device=self.device).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], resolution, device=self.device).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], resolution, device=self.device).split(N)

        u = torch.zeros([resolution, resolution, resolution], device=self.device)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(self.device)
                        val = self.sdf_func(pts).reshape(len(xs), len(ys), len(zs)).detach()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val         
        return u
    
    def extract_geometry(self, verbose=True):
        u = self.extract_fields()
        cleanup()
        vertices, faces = cumcubes.marching_cubes(
            u, 
            self.threshold, 
            verbose=verbose, 
            cpu=not self.use_cuda
            )
        if not self.use_cuda:
            self.bound_min = self.bound_min.cpu()
            self.bound_max = self.bound_max.cpu()
            vertices = vertices.to(torch.float32)
            faces = faces.to(torch.int32)
        vertices = vertices / (self.resolution - 1.0) * (self.bound_max - self.bound_min)[None, :] + self.bound_min[None, :]
        return vertices, faces
    
    def extract_color(self, vertices):
        out = self.model.geometry_network(vertices, with_feature=True, with_grad=True, with_laplace=False)
        feats, normal = out['features'], out['normal']
        rgbs = self.model.color_network(-normal, feats, points=vertices, normal=normal).detach().cpu() * 255
        return rgbs

    @torch.no_grad()
    def save_mesh(self, filename):
        vertices, faces = self.extract_geometry()
        color = None
        if self.export_color:
            color_blocks = []
            for vertices_block in vertices.split(self.block_size ** 3):
                # query vertices color 
                rgbs = self.color_func(vertices_block)
                color_blocks.append(rgbs)
            color = torch.concatenate(color_blocks)
            print('Add color to vertices.')
        # transform vertices from aabb to world
        if self.scale is not None:
            vertices = vertices * self.scale + self.center
            print('Transform vertices from aabb to world.')
        cumcubes.save_mesh(vertices, faces, color, filename=filename)



if __name__ == "__main__":
    pass