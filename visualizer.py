# Use pytorch3d renderer for visualize mesh
# https://github.com/cr00z/virtual_tryon


import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
import cv2


class Visualizer:
    def __init__(self):
        self.mesh_color = torch.Tensor([[[0.65098039, 0.74117647, 0.85882353]]])
        self.input_size = 1920
        self.render_size = 700
        self.device = torch.device('cuda') if torch.cuda.is_available()\
            else torch.device('cpu')
        lights = PointLights(
            ambient_color=[[1, 1, 1]],
            diffuse_color=[[1, 1, 1]],
            device=self.device, location=[[1, 1, -30]])
        camera = FoVOrthographicCameras(
            device=self.device,
            znear=0.1,
            zfar=10.0,
            max_y=1.0,
            min_y=-1.0,
            max_x=1.0,
            min_x=-1.0,
            scale_xyz=((1.0, 1.0, 1.0),),
        )
        raster_settings = RasterizationSettings(
            image_size=self.render_size, blur_radius=0, faces_per_pixel=1,
        )
        blend_params = BlendParams(
            sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=lights,
                blend_params=blend_params
            )
        )

    def visualize(self, img_original_bgr, bbox, mesh_data):
        verts = mesh_data.v
        faces = torch.Tensor(mesh_data.f.astype(np.float32))

        # render predicted meshes

        # bbox for verts
        x0 = int(np.min(verts[:, 0]))
        x1 = int(np.max(verts[:, 0]))
        y0 = int(np.min(verts[:, 1]))
        y1 = int(np.max(verts[:, 1]))
        width = x1 - x0
        height = y1 - y0

        # padding the tight bbox
        margin = int(max(width, height) * 0.1)
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(self.input_size, x1 + margin)
        y1 = min(self.input_size, y1 + margin)

        # move verts to be in the bbox
        verts[:, 0] -= x0
        verts[:, 1] -= y0

        # normalize verts to (-1, 1)
        bbox_size = max(y1 - y0, x1 - x0)
        half_size = bbox_size / 2
        verts[:, 0] = (verts[:, 0] - half_size) / half_size
        verts[:, 1] = (verts[:, 1] - half_size) / half_size

        # the coords of pytorch-3d is (1, 1) for upper-left
        # and (-1, -1) for lower-right
        # so need to multiple minus for vertices
        verts[:, :2] *= -1

        # shift verts along the z-axis
        verts[:, 2] /= 112
        verts[:, 2] += 5

        verts = torch.Tensor(verts)

        tex = cv2.imread(mesh_data.texture_filepath)[:, :, ::-1]    # RGB -> BGR
        tex = torch.Tensor(tex / 255.)[None]
        verts_uvs = torch.Tensor(mesh_data.vt.astype(np.float32))[None]
        faces_uvs = torch.LongTensor(mesh_data.ft.astype(np.int32))[None]

        textures = TexturesUV(
            maps=tex,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs
        )

        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        ).to(self.device)

        rend_img = self.renderer(mesh)[0].cpu().numpy()

        # blending rendered mesh with background image

        scale_ratio = self.render_size / bbox_size
        img_size_new = int(self.input_size * scale_ratio)

        x0 = max(int(x0 * scale_ratio), 0)
        y0 = max(int(y0 * scale_ratio), 0)
        x1 = min(int(x1 * scale_ratio), img_size_new)
        y1 = min(int(y1 * scale_ratio), img_size_new)

        h0 = min(y1 - y0, self.render_size)
        w0 = min(x1 - x0, self.render_size)

        y1 = y0 + h0
        x1 = x0 + w0

        rend_img_new = np.zeros((img_size_new, img_size_new, 4))
        rend_img_new[y0:y1, x0:x1, :] = rend_img[:h0, :w0, :]
        rend_img = rend_img_new

        alpha = np.zeros((img_size_new, img_size_new, 1), dtype=np.uint8)
        alpha[rend_img[:, :, 3:4] > 0] = 1

        rend_img = rend_img[:, :, :3]
        max_color = rend_img.max()
        rend_img *= 255 / max_color  # Make sure <1.0
        rend_img = rend_img[:, :, ::-1].astype(np.uint8)

        bg_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        h, w = img_original_bgr.shape[:2]
        bg_img[:h, :w, :] = img_original_bgr

        bg_img = cv2.resize(bg_img, (img_size_new, img_size_new))
        res_img = alpha * rend_img + (1 - alpha) * bg_img
        res_img = cv2.resize(res_img, (self.input_size, self.input_size))
        res_img = res_img[:h, :w, :]

        return res_img
