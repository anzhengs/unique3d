import os
import torch
from scripts.sd_model_zoo import load_common_sd15_pipe
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline


class MyModelZoo:
    _pipe_disney_controlnet_lineart_ipadapter_i2i: StableDiffusionControlNetImg2ImgPipeline = None
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model = os.path.join(root_dir, "ckpt/v1-5-pruned-emaonly.safetensors")  # 本地模型路径


    def __init__(self, base_model=None) -> None:
        if base_model is not None:
            self.base_model = base_model

    @property
    def pipe_disney_controlnet_tile_ipadapter_i2i(self):
        return self._pipe_disney_controlnet_lineart_ipadapter_i2i
    
    def init_models(self):  # 将方法移到类内部
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        controlnet_path = os.path.join(root_dir, "ckpt/controlnet-tile")
        
        self._pipe_disney_controlnet_lineart_ipadapter_i2i = load_common_sd15_pipe(
            base_model=self.base_model, 
            ip_adapter=True, 
            plus_model=False, 
            controlnet=controlnet_path,  # 使用绝对路径
            pipeline_class=StableDiffusionControlNetImg2ImgPipeline
        )

model_zoo = MyModelZoo()