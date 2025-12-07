# modified from https://github.com/Profactor/continuous-remeshing
import nvdiffrast.torch as dr
import torch
from typing import Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    TexturesVertex,
    MeshRasterizer,
    BlendParams,
    FoVOrthographicCameras,
    look_at_view_transform,
    hard_rgb_blend,
)

# 定义device变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

# 替换为CUDA上下文
glctx = dr.RasterizeCudaContext(device=device)

class NormalsRenderer:
    # 修改类型注解
    _glctx: dr.RasterizeCudaContext = None

    def __init__(
            self,
            mv: torch.Tensor,  # C,4,4
            proj: torch.Tensor,  # C,4,4
            image_size: Tuple[int, int],
            mvp=None,
            device=None,
    ):
        if mvp is None:
            self._mvp = proj @ mv  # C,4,4
        else:
            self._mvp = mvp
        self._image_size = image_size
        self._glctx = glctx
        _warmup(self._glctx, device)

    def render(self,
               vertices: torch.Tensor,  # V,3 float
               normals: torch.Tensor,  # V,3 float   in [-1, 1]
               faces: torch.Tensor,  # F,3 long
               ) -> torch.Tensor:  # C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)
        vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
        rast_out, _ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size)
        vert_col = (normals + 1) / 2  # V,3 转换到 [0,1]
        col, _ = dr.interpolate(vert_col, rast_out, faces)  # C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
        col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
        return col  # C,H,W,4

class VertexColorShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)

def render_mesh_vertex_color(mesh, cameras, H, W, blur_radius=0.0, faces_per_pixel=1, 
                            dtype=torch.float32, bkgd=(0.0, 0.0, 0.0)):
    if len(mesh) != len(cameras):
        if len(cameras) % len(mesh) == 0:
            mesh = mesh.extend(len(cameras))
        else:
            raise NotImplementedError()

    # render requires everything in float16 or float32
    input_dtype = dtype
    blend_params = BlendParams(1e-4, 1e-4, bkgd)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True,
        bin_size=None,
        max_faces_per_bin=None,
    )
    # Create a renderer by composing a rasterizer and a shader
    # We simply render vertex colors through the custom VertexColorShader (no lighting)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=VertexColorShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params
        )
    )

    # render RGB and depth, get mask
    with torch.autocast(dtype=input_dtype, device_type=torch.device(device).type):
        images, _ = renderer(mesh)
    return images   # BHW4

class Pytorch3DNormalsRenderer:  # 100 times slower!!!
    def __init__(self, cameras, image_size, device):
        self.cameras = cameras.to(device)
        self._image_size = image_size
        self.device = device

    def render(self,
               vertices: torch.Tensor,  # V,3 float
               normals: torch.Tensor,  # V,3 float   in [-1, 1]
               faces: torch.Tensor,  # F,3 long
               ) -> torch.Tensor:  # C,H,W,4
        # 将法线转换为颜色 [0,1]
        vert_col = (normals + 1) / 2
        mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[vert_col]))
        return render_mesh_vertex_color(mesh, self.cameras, self._image_size[0], self._image_size[1], 
                                       dtype=torch.float32, bkgd=(0.0, 0.0, 0.0))

def save_tensor_to_img(tensor, save_dir):
    import os
    from PIL import Image
    import numpy as np
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, img in enumerate(tensor):
        img = img[..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f"{idx}.png"))

# 辅助函数：创建正交相机
def make_star_cameras_orthographic_py3d(azimuths, device="cuda", distance=1.0, elevation=0):
    cameras = []
    for az in azimuths:
        R, T = look_at_view_transform(distance=distance, elevation=elevation, azimuth=az, device=device)
        cameras.append(FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01, zfar=10.0))
    return cameras

def make_star_cameras_orthographic(num_views, scale=1):
    """创建正交投影矩阵"""
    mv = []
    proj = []
    
    for i in range(num_views):
        # 视图矩阵（不同角度）
        angle = i * 90
        if angle == 0:
            R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        elif angle == 90:
            R = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        elif angle == 180:
            R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
        else:  # 270
            R = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float32)
        
        T = torch.tensor([[0, 0, -2]], dtype=torch.float32)
        mv_matrix = torch.eye(4, dtype=torch.float32)
        mv_matrix[:3, :3] = R
        mv_matrix[:3, 3] = T
        
        # 投影矩阵
        proj_matrix = torch.eye(4, dtype=torch.float32)
        proj_matrix[0, 0] = 1/scale
        proj_matrix[1, 1] = 1/scale
        proj_matrix[2, 2] = 1/scale
        
        mv.append(mv_matrix)
        proj.append(proj_matrix)
    
    return torch.stack(mv), torch.stack(proj)

if __name__ == "__main__":
    # 创建相机
    cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device=device, distance=2.0)
    mv, proj = make_star_cameras_orthographic(4, 1)
    
    # 确保矩阵在正确设备上
    mv = mv.to(device)
    proj = proj.to(device)
    
    resolution = 512  # 降低分辨率加快测试
    renderer1 = NormalsRenderer(mv, proj, [resolution, resolution], device=device)
    renderer2 = Pytorch3DNormalsRenderer(cameras, [resolution, resolution], device=device)
    
    # 创建测试四面体数据
    vertices = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[1,0,0]], device=device, dtype=torch.float32)
    normals = torch.tensor([[-1,-1,-1],[1,-1,-1],[-1,-1,1],[-1,1,-1]], device=device, dtype=torch.float32)
    faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], device=device, dtype=torch.int64)

    # 性能测试
    import time
    
    print("=== 渲染性能测试 ===")
    t0 = time.time()
    r1 = renderer1.render(vertices, normals, faces)
    print(f"nvdiffrast CUDA渲染时间: {time.time() - t0:.4f}秒")

    t0 = time.time()
    r2 = renderer2.render(vertices, normals, faces)
    print(f"PyTorch3D渲染时间: {time.time() - t0:.4f}秒")

    # 结果比较
    print("\n=== 渲染结果比较 ===")
    for i in range(4):
        diff_mean = (r1[i] - r2[i]).abs().mean().item()
        total_mean = (r1[i] + r2[i]).abs().mean().item()
        print(f"视角{i}: 平均差异={diff_mean:.6f}, 平均强度={total_mean:.6f}")

    # 保存渲染结果
    save_tensor_to_img(r1, "./output_nvdiffrast/")
    save_tensor_to_img(r2, "./output_pytorch3d/")
    print("\n渲染结果已保存到 output_nvdiffrast/ 和 output_pytorch3d/ 目录")
