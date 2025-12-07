from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import torch
from copy import deepcopy
import os
import sys

ENABLE_CPU_CACHE = False
DEFAULT_BASE_MODEL = "/home/tjut_shianzheng/unique3d/Unique3D-main/ckpt/v1-5-pruned-emaonly.safetensors"
# 添加本地模型路径配置
LOCAL_IMAGE_VARIATIONS_PATH = "/home/tjut_shianzheng/unique3d/Unique3D-main/ckpt/sd-image-variations"
LOCAL_IP_ADAPTER_PATH = "/home/tjut_shianzheng/unique3d/Unique3D-main/ckpt/IP-Adapter"  # 建议下载IP-Adapter到本地

cached_models = {}  # cache for models to avoid repeated loading, key is model name

def cache_model(func):
    def wrapper(*args, **kwargs):
        if ENABLE_CPU_CACHE:
            model_name = func.__name__ + str(args) + str(kwargs)
            if model_name not in cached_models:
                cached_models[model_name] = func(*args, **kwargs)
            return cached_models[model_name]
        else:
            return func(*args, **kwargs)
    return wrapper

def copied_cache_model(func):
    def wrapper(*args, **kwargs):
        if ENABLE_CPU_CACHE:
            model_name = func.__name__ + str(args) + str(kwargs)
            if model_name not in cached_models:
                cached_models[model_name] = func(*args, **kwargs)
            return deepcopy(cached_models[model_name])
        else:
            return func(*args, **kwargs)
    return wrapper

def model_from_ckpt_or_pretrained(ckpt_or_pretrained, model_cls, **kwargs):
    # 处理sd-image-variations特殊情况
    if ckpt_or_pretrained == "lambdalabs/sd-image-variations-diffusers" and os.path.exists(LOCAL_IMAGE_VARIATIONS_PATH):
        ckpt_or_pretrained = LOCAL_IMAGE_VARIATIONS_PATH
    
    # 对于sd-image-variations，禁用safetensors检查
    if "sd-image-variations" in str(ckpt_or_pretrained).lower():
        kwargs["use_safetensors"] = False
    
    # 检查是否为本地文件（单文件：.safetensors/.bin）
    if os.path.isfile(ckpt_or_pretrained):
        # 本地单文件，使用from_single_file加载
        pipe = model_cls.from_single_file(
            ckpt_or_pretrained,
            **kwargs
        )
    # 检查是否为本地目录
    elif os.path.isdir(ckpt_or_pretrained):
        # 本地目录，使用from_pretrained加载
        pipe = model_cls.from_pretrained(
            ckpt_or_pretrained,
            local_files_only=True,
            **kwargs
        )
    else:
        # Hugging Face模型库
        pipe = model_cls.from_pretrained(
            ckpt_or_pretrained,
            local_files_only=False,
            **kwargs
        )
    return pipe

@copied_cache_model
def load_base_model_components(base_model=DEFAULT_BASE_MODEL, torch_dtype=torch.float16):
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        requires_safety_checker=False, 
        safety_checker=None,
    )
    # 直接加载完整pipeline，再提取components
    pipe: StableDiffusionPipeline = model_from_ckpt_or_pretrained(
        base_model,
        StableDiffusionPipeline,
        **model_kwargs
    )
    pipe.to("cuda")  # 修改为直接加载到GPU，避免CPU警告
    # 确保components包含所有必要组件
    components = {
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,
        "scheduler": pipe.scheduler,
        "safety_checker": pipe.safety_checker,
        "feature_extractor": pipe.feature_extractor,
    }
    return components

@cache_model
def load_controlnet(controlnet_path, torch_dtype=torch.float16):
    # 如果是本地路径，使用local_files_only=True
    if os.path.exists(controlnet_path):
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, 
            local_files_only=True, 
            torch_dtype=torch_dtype
        )
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, 
            torch_dtype=torch_dtype
        )
    return controlnet

@cache_model
def load_image_encoder():
    # 优先使用本地IP-Adapter模型
    image_encoder_path = LOCAL_IP_ADAPTER_PATH if os.path.exists(LOCAL_IP_ADAPTER_PATH) else "h94/IP-Adapter"
    local_files = os.path.exists(LOCAL_IP_ADAPTER_PATH)
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_path,
        subfolder="models/image_encoder",
        local_files_only=local_files,
        torch_dtype=torch.float16,
    )
    return image_encoder

def load_common_sd15_pipe(base_model=DEFAULT_BASE_MODEL, device="cuda", controlnet=None, ip_adapter=False, plus_model=True, torch_dtype=torch.float16, model_cpu_offload_seq=None, enable_sequential_cpu_offload=False, vae_slicing=False, pipeline_class=None,** kwargs):
    model_kwargs = dict(
        torch_dtype=torch_dtype, 
        device_map=device,
        requires_safety_checker=False, 
        safety_checker=None,
    )
    components = load_base_model_components(base_model=base_model, torch_dtype=torch_dtype)
    model_kwargs.update(components)
    model_kwargs.update(kwargs)
    
    if controlnet is not None:
        if isinstance(controlnet, list):
            controlnet = [load_controlnet(controlnet_path, torch_dtype=torch_dtype) for controlnet_path in controlnet]
        else:
            controlnet = load_controlnet(controlnet, torch_dtype=torch_dtype)
        model_kwargs.update(controlnet=controlnet)
    
    if pipeline_class is None:
        if controlnet is not None:
            pipeline_class = StableDiffusionControlNetPipeline
        else:
            pipeline_class = StableDiffusionPipeline
    
    pipe: StableDiffusionPipeline = model_from_ckpt_or_pretrained(
        base_model,
        pipeline_class,** model_kwargs
    )

    if ip_adapter:
        image_encoder = load_image_encoder()
        pipe.image_encoder = image_encoder
        
        # IP-Adapter本地加载支持
        ip_adapter_path = LOCAL_IP_ADAPTER_PATH if os.path.exists(LOCAL_IP_ADAPTER_PATH) else "h94/IP-Adapter"
        local_files = os.path.exists(LOCAL_IP_ADAPTER_PATH)
        
        if plus_model:
            pipe.load_ip_adapter(
                ip_adapter_path, 
                subfolder="models", 
                weight_name="ip-adapter-plus_sd15.safetensors",
                local_files_only=local_files
            )
        else:
            pipe.load_ip_adapter(
                ip_adapter_path, 
                subfolder="models", 
                weight_name="ip-adapter_sd15.safetensors",
                local_files_only=local_files
            )
        pipe.set_ip_adapter_scale(1.0)
    else:
        if hasattr(pipe, 'unload_ip_adapter'):
            pipe.unload_ip_adapter()
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if model_cpu_offload_seq is None:
        if isinstance(pipe, StableDiffusionControlNetPipeline):
            pipe.model_cpu_offload_seq = "text_encoder->controlnet->unet->vae"
        elif isinstance(pipe, StableDiffusionControlNetImg2ImgPipeline):
            pipe.model_cpu_offload_seq = "text_encoder->controlnet->vae->unet->vae"
    else:
        pipe.model_cpu_offload_seq = model_cpu_offload_seq
    
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to("cuda")  # 确保模型加载到GPU
    
    if vae_slicing:
        pipe.enable_vae_slicing()
        
    import gc
    gc.collect()
    return pipe