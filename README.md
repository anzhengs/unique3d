# Unique3D

一个强大的3D计算机视觉与深度学习项目，专注于3D重建、渲染和生成。

## ✨ 特性

- 🚀 高效的3D数据处理管道
- 🎨 高质量3D渲染与可视化
- 🤖 基于深度学习的3D模型生成
- 🔧 模块化设计，易于扩展
- 📊 评估工具

## 📁 项目结构
unique3d/
├── Unique3D-main/ # 主项目核心代码
│ ├── src/ # 源代码
│ ├── configs/ # 配置文件
│ ├── models/ # 模型定义
│ └── utils/ # 工具函数
├── nvdiffrast/ # NVIDIA高性能渲染库
├── requirements.txt # 项目依赖
├── uni_requirements.txt # 额外依赖
└── README.md # 项目说明

## 个人运行代码
(uni) tjut_shianzheng@user-NF5468M5:~/unique3d/Unique3D-main$ CUDA_VISIBLE_DEVICES=1 \
> HF_HOME=/data/sazdata/.cache/huggingface \
> GRADIO_TEMP_DIR=$(pwd)/app/gradio_temp \
> python app/gradio_local.py --port 7860
