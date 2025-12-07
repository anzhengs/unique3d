from omegaconf import DictConfig, OmegaConf
import os  # 新增：处理路径拼接


def parse_structured(fields, cfg) -> DictConfig:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def load_config(fields, config, extras=None):
    if extras is not None:
        print("Warning! extra parameter in cli is not verified, may cause erros.")
    if isinstance(config, str):
        cfg = OmegaConf.load(config)
    elif isinstance(config, dict):
        cfg = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        cfg = config
    else:
        raise NotImplementedError(f"Unsupported config type {type(config)}")
    
    # 新增：替换模型路径为本地 ckpt（若配置中有模型路径字段）
    if "model_path" in cfg and cfg.model_path == "/home/tjut_shianzheng/unique3d/Unique3D-main/ckpt/sd-image-variations-diffusers":
        cfg.model_path = "/home/tjut_shianzheng/unique3d/Unique3D-main/ckpt"
    # 或直接指定 SD 模型路径字段（根据配置文件实际字段名调整）
    if "sd_model_path" in cfg:
        cfg.sd_model_path = os.path.abspath(cfg.sd_model_path)  # 转为绝对路径
    
    if extras is not None:
        cli_conf = OmegaConf.from_cli(extras)
        cfg = OmegaConf.merge(cfg, cli_conf)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return parse_structured(fields, cfg)