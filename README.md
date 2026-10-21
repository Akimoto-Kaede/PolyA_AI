# PolyA_AI

PAL-AI 复现实验与拟南芥叶片（`ara_leaf`）三代测序数据上的 poly(A) 尾长建模代码，包括中位数预测、单读段回归、尾长分布预测及 DNABERT-2 微调。

本仓库汇集 IPF 与 Cryo-EM 上的历史实验代码。当前版本用于归档与后续整理：保留各实验实现、参数与服务器路径，不代表已经从干净环境完成端到端复现，也不声称全部实验均已达到论文结果。

## 代码导航

| 目录 / 文件 | 内容 | 来源 |
| --- | --- | --- |
| `PAL-AI/` | TensorFlow / PyTorch PAL-AI 实验及 PBS 作业示例 | IPF |
| `polya_med/` | poly(A) 中位数预测版本 | IPF |
| `pal_polya_med/` | 中位数模型训练、独立测试及 LSF 脚本 | Cryo-EM |
| `extremum_test/` | 极端尾长样本实验 | Cryo-EM |
| `polya_dist/1_resgrutransformer/` | 残差卷积、GRU、Transformer 分布建模及 WAE 版本 | Cryo-EM |
| `polya_dist/ath/` | 拟南芥分布预测实验 | Cryo-EM |
| `polya_dist/embedding/wae/` | Wasserstein autoencoder 实验 | Cryo-EM |
| `single_read/ara_leaf/` | 叶片单读段尾长回归模型 | Cryo-EM |
| `dnabert2/polya_single_read/` | DNABERT-2 微调与 gene_200 数据划分实验 | Cryo-EM |
| `ai.ipynb`, `ai_polyA.ipynb` | 数据处理、探索和模型实验 notebook；已清除运行输出 | IPF |

## 使用前准备

1. 阅读 [环境说明](docs/ENVIRONMENTS.md)，为对应实验选择 TensorFlow/Keras 或 PyTorch 环境。
2. 根据 [数据与路径说明](docs/DATA_AND_PATHS.md) 准备输入、预训练权重及输出目录。
3. 调整脚本中的服务器绝对路径。当前脚本尚未统一成命令行接口，直接运行会开始实际训练或评估；导入模块也可能执行训练。
4. 先在相应集群的测试作业中验证依赖与输入，再提交完整训练。

以下仅为入口示例，**需先完成以上准备**：

```bash
python PAL-AI/pal-ai.py
python single_read/ara_leaf/ath_gt_single_read.py
python dnabert2/polya_single_read/dnabert2_gene_200.py
```

`*.sh` 中的 `#BSUB` 参数适用于原 Cryo-EM LSF 环境；`*.pbs` 属于原 IPF 作业示例。队列、节点、CPU/GPU 配额、日志路径和 Conda 激活路径均需按运行环境修改。

## 数据与结果范围

本次提交仅包含代码、清理后的 notebook、作业脚本、DNABERT-2 配置/tokenizer/实现文件和说明文档。原始测序数据、处理后的训练数组/标签、模型权重、预测表及训练图不在仓库中。未提供可复核的训练结果汇总，故不在 README 报告精度数值。

DNABERT-2 目录中的 `data/dnabert2_117m` 是第三方模型实现及配置快照，**不包含完整预训练权重**。其原始 `LICENSE` 和 `README.md` 保留。

## 来源与验证

- [来源和 SHA256 清单](docs/source_manifest.tsv) 记录每个导入文件的来源及整理前后哈希。同路径检查未发现 IPF/Cryo-EM 文件冲突。
- 仅 notebook 输出与执行元数据被清除；其余导入文件保持原始内容。
- 本次执行了 Python 语法解析、shell 语法检查、notebook JSON/输出检查及提交内容检查，未启动模型训练。
- Cryo-EM 副本来自按文件类型筛选的复制，单文件上限为 20 MB；未包括的其他脚本类型需后续另行核对。

## 引用与第三方来源

本项目参考 PAL-AI 方法，并开展独立的拟南芥数据实验；不是 PAL-AI 或 DNABERT-2 官方仓库。

- PAL-AI：[coffeebond/PAL-AI](https://github.com/coffeebond/PAL-AI)
- Xiang, K. & Bartel, D. P. *PAL-AI reveals genetic determinants that control poly(A)-tail length during oocyte maturation, with relevance to human fertility.* Nature Communications 16, 7079 (2025). [DOI](https://doi.org/10.1038/s41467-025-62171-5)
- DNABERT-2：[zhihan1996/DNABERT_2](https://github.com/zhihan1996/DNABERT_2)，[模型](https://huggingface.co/zhihan1996/DNABERT-2-117M)。第三方文件许可见其目录内的 `LICENSE`。

项目自有代码暂未指定统一开源许可证；第三方文件沿用各自声明，不将第三方许可扩展为整个仓库的许可。
