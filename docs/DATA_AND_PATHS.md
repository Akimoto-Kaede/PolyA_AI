# 数据与路径

## 原始目录

- IPF：`/DataZhai/zhangyt/project/ai`。
- IPF 代码中还保留 `/public/home/zhangyt/project/ai` 历史路径；需确认其是否为有效映射。
- Cryo-EM：`/scem/work/zhangyt/project/ai`。
- 当前 Git 工作目录：`/DataZhai/zhangyt/git_repo/PolyA_AI`。

脚本中的路径保持原样，因此从仓库目录启动并不意味着会读写仓库下的相对目录。运行前检查所有输入、模型和输出路径，避免将新实验写入历史结果目录。

## 输入类别

- PAL-AI / 中位数 / 极值：`x_input.npy`、`y_input.csv`；具体 shape、列名及数据截取以对应脚本为准。PAL-AI 中存在 `(1056, 4)` 输入结构。
- 单读段和分布模型：实验目录中的 NumPy 数组、CSV 标签或分布数据。各模型的序列长度、输出维数、归一化和样本划分不同，不可直接混用。
- DNABERT-2 普通版本：`ara_leaf_3000_chr1235.csv`，读取 `read_seq` 与 `log1p_scaled` 列。
- DNABERT-2 gene_200 版本：`gene_200/train_data_leaf.250820.csv` 和 `gene_200/eval_data_leaf.250820.csv`。
- DNABERT-2 通过本地路径及 `local_files_only=True` 加载，须另外准备匹配的完整预训练权重，配置/tokenizer 文件本身不足以运行。

现有 PAL-AI 输入还包括 `XL_utr3.fa` 以及 `XL_oo_pp_*` 表格；这些输入在本次代码提交中省略。原始服务器上的文件未更改。

## 运行限制

1. 事先建立脚本预期的 `models`、`model`、`prediction`、`result`、`loss_plot`、`scatter_plot` 等目录，具体取决于实验。
2. 不同脚本的训练/验证切分及重复训练选模逻辑各异；评估方法尚未在本次归档中统一或审计。
3. `pal_polya_med/pal_test.py` 保留了 `pal_polya_med` 与 `polya_med` 混用的路径，运行前需要核对。
4. `.gitignore` 排除常见数据、权重、输出文件；不要用 `git add -f` 将大数据或凭据加入仓库。
