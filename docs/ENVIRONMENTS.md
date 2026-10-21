# 环境说明

本次整理未安装依赖或运行训练。以下是代码依赖梳理及 IPF 现有环境的元数据快照，不能视为 Cryo-EM 原训练环境的锁定文件。

## 主要依赖

- 通用：NumPy、pandas、SciPy、scikit-learn、matplotlib、joblib。
- TensorFlow 实验：TensorFlow、Keras；分布损失部分使用 POT（导入名 `ot`）。
- PyTorch PAL-AI：PyTorch。
- DNABERT-2：PyTorch、transformers、einops；第三方 flash-attention 实现使用 Triton，需匹配 CUDA/GPU 与相应版本。
- Notebook：Jupyter。不同 notebook/实验可能需要不同环境。

## 已观察到的 IPF 环境

| 环境 | 从 Conda 元数据读取的部分版本 |
| --- | --- |
| `ai` | Python 3.11.5、TensorFlow/Keras 2.12.0、PyTorch 2.5.1、NumPy 1.23.5 |
| `tf` | Python 3.10.16、TensorFlow 2.18.0、NumPy 1.24.0、POT 0.9.5 |
| `torch` | Python 3.10.0、NumPy 2.2.5；PyTorch/transformers 实际 pip 版本未核实 |

上述包可能混合 Conda/pip 安装，表格不代表已验证兼容性。部分历史脚本依赖旧 Keras API，而其他脚本使用 `keras.saving` 等较新 API，不建议将所有实验强行安装到同一套最新版环境中。复现实验时应记录实际 Python、依赖、CUDA/cuDNN、GPU 和随机种子。
