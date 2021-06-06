import unittest
from pathlib import Path
import os
import numpy as np
import torch
import pickle as pkl
from PIL import Image
from pytorch3d.renderer.cameras import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    SoftSilhouetteShader,
)

from pytorch3d.structures.meshes import Meshes
import torch.nn as nn

def perspective_projection_new(f, c, w, h, near=0.1, far=10.):

    f = 0.5 * (f[0] + f[1])
    pixel_center_offset = 0.5
    right = (w - (c[0] + pixel_center_offset)) * (near / f)
    left = -(c[0] + pixel_center_offset) * (near / f)
    top = (c[1] + pixel_center_offset) * (near / f)
    bottom = -(h - c[1] + pixel_center_offset) * (near / f)

    elements = [
        [3 * near / (right - left), 0., (right + left) / (right - left), 0.],
        [0., 3 * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
        [0., 0., far / (far - near), - far * near / (far - near)],
        [0., 0., 1., 0.]
    ]

    return torch.transpose(torch.tensor(elements, dtype=torch.float32), 1, -1)



def render(verts, faces, frame_dim, near, far, device):

    faces1 = torch.LongTensor(faces[np.newaxis, :,:].astype(np.int32)).to(device)
    mesh = Meshes(verts=verts, faces=faces1)

    blend_params = BlendParams(sigma=1e-10, gamma=0)
    raster_settings = RasterizationSettings(
        image_size=1080,
        blur_radius=np.log(1.0 / 1e-10 - 1.0) * blend_params.sigma,
        faces_per_pixel=10,
    )

    # Init rasterizer settings
    R, T = look_at_view_transform(1, 0, 0)

    camera_f = [frame_dim, frame_dim]
    camera_c = [frame_dim/2, frame_dim/2]

    KK = perspective_projection_new(camera_f, camera_c, frame_dim, frame_dim, near=0.1, far=10.)
    R, T = look_at_view_transform(1, 0, 0)
    #cameras = FoVPerspectiveCameras(K=KK[None, :, :], device=device, R=R, T=T)


    cameras = FoVPerspectiveCameras(znear=0.1, zfar=10, aspect_ratio=1.0, device=device, fov=36, R=R, T=T)

    # Init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    images = renderer(mesh)
    alpha = images[0, ..., 3].squeeze()

    return alpha


