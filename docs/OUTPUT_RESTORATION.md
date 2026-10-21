# Notebook 输出恢复记录

本次从 IPF 原始 notebook 恢复图和可读统计输出，未重新执行训练或计算指标。原始服务器 notebook 未改动。

保留全部 PNG 图、统计/标量输出及模型结构摘要；移除错误输出、训练进度、设备诊断日志和部分原始序列记录预览。保留结果的执行计数供追溯，不代表全 notebook 顺序运行。新增开头说明与原始单元元数据，代码原样保留。

- `ai.ipynb`：保留 30 张 PNG、68 项输出；移除 539 项日志/错误/预览输出。
- `ai_polyA.ipynb`：保留 4 张 PNG、17 项输出；移除 10 项日志/错误/预览输出。

图片清单见 `figures/manifest.json`，notebook 的源文件和整理后哈希见 `source_manifest.tsv`。
