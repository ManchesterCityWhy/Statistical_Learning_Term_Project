# T1 竞赛实验进展（2026-11-24）

[逐次实验评分与运行归档](archive/2026-11-24/README.md) · [中期报告公开版](../reports/T1_中期报告_公开版.pdf)

每轮用同一组参考细胞和同一组 E10.5 评价细胞比较候选，抽样种子为 0、31、101。生成开发预测时，竞赛数据的拟合阶段限于 E8.5 和 E9.5，E10.5 用于开发评价和候选选择。不同抽样种子复用同一阶段的数据，不能视为独立胚胎。

| 方法 | 种子 0 | 种子 31 | 种子 101 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| 分类型保零 + GEO bulk 发育趋势 | 52.269 | 53.201 | 52.996 | 52.822 |
| 分类型保零 + E14.5 单细胞输运 | 52.079 | 52.526 | 52.940 | 52.515 |
| 细胞状态重加权 + 分类型保零 | 51.083 | 53.344 | 52.423 | 52.283 |
| 分类型保零、归一化 | 50.451 | 51.268 | 51.077 | 50.932 |
| 保零 PCA10 | 49.811 | 50.250 | 50.196 | 50.086 |
| 复制最近阶段 | 50.000 | 50.000 | 50.000 | 50.000 |

以上为本地开发分数，经本轮抽样各自的 floor 和 ceiling 校准。bulk 组合的平均 MMD 为 0.07032，复制基线为 0.08500。单细胞输运组合的平均 CSS 为 0.00387，复制基线为 0.00571。三组领先模型的平均 DES 仍为负，差异表达恢复是主要缺口。

## 数据和三种候选

- 竞赛输入为官方下载的 T1 E8.5、E9.5 和 P3 已授权的 E10.5。竞赛数据由 UC San Diego 的 Neil Chi 团队生成，并由 Virtual Embryo Challenge 发布。最终预测可用 E10.5，测试 E12.5 真值始终保留在官方。 [数据](https://virtualembryo.ai/challenge/data) · [规则](https://virtualembryo.ai/challenge/rules)
- `bulk_type_075` 使用 GSE223237 的 3 个 E9.5 样本（GSM6943750–52）和 3 个 E14.5 样本（GSM6943761–63）。只下载上述 6 个单独的计数文件，另用 GSM6211349 的基因注释表做 Ensembl ID 转换。先按组进行保零平移，再迁移六个 bulk 重复估计出的基因级变化，并恢复每细胞 10,000 的归一化尺度。 [GEO 研究](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223237) · [基因注释来源](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6211349)
- `geo_type_075` 使用 GSE150817 的 CELLxGENE 版本。仅从原始计数中取 E14.5、正常、心脏的 8,889 个细胞。先做分类型保零平移，再按跨平台的细胞状态匹配结果做部分输运，联合调整非零项和表达幅度。 [原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13602375/) · [CELLxGENE 数据](https://cellxgene.cziscience.com/collections/613f5480-4957-4f80-b804-0e2b85ac454c)
- `type_keepzero_weighted_norm` 仅用竞赛数据。在共同变异基因的 PCA 表示中拟合早、晚阶段的分类器，用经截断的分类 log-odds 给晚阶段细胞重加权。对同名细胞群体用观测到的均值差做保零平移，最后重新归一化。

## 复现

先按 [T1 运行说明](../t1_pipeline/README.md) 初始化环境、同步公开基因清单，并把受限竞赛数据放到 `data/T1/`。

外部数据放到以下本地目录。GEO 文件用单个 GSM 样本条目里的补充文件名。CELLxGENE 选择该集合的 H5AD 格式，下载后重命名。

```text
data/external/GSE223237/GSM6943750_E9_Rep1_counts.txt.gz
data/external/GSE223237/GSM6943751_E9_Rep2_counts.txt.gz
data/external/GSE223237/GSM6943752_E9_Rep3_counts.txt.gz
data/external/GSE223237/GSM6943761_E14_Rep1_counts.txt.gz
data/external/GSE223237/GSM6943762_E14_Rep2_counts.txt.gz
data/external/GSE223237/GSM6943763_E14_Rep3_counts.txt.gz
data/external/GSM6211349/GSM6211349_E14_genes.tsv.gz
data/external/GSE150817/perinatal_heart.h5ad
```

```bash
.venv/bin/python -m t1_pipeline benchmark --config configs/t1_competitive.json --output runs/competitive_screen
.venv/bin/python -m t1_pipeline benchmark --config configs/t1_competitive_verify.json --output runs/competitive_verify_0
.venv/bin/python -m experiments.bulk_trend benchmark --output runs/competitive_bulk --seeds 0 31 101
.venv/bin/python -m experiments.geo_transport benchmark --output runs/competitive_geo_transport --seeds 0 31 101
```

重跑另外两轮内部对照时，将 `t1_competitive_verify.json` 分别复制到两个新配置，将 `seeds` 改为 `[31]` 或 `[101]`，将 `evaluation_seed` 改为相同的值，指定各自的输出目录。

当前两种主要候选的 E12.5 文件均由种子 0 生成。下面的命令会重建文件并执行本地格式检查。

```bash
.venv/bin/python -m experiments.bulk_trend export --candidate bulk_type_075 --seeds 0 --output predictions/e125_competitive_bulk
.venv/bin/python -m experiments.geo_transport export --candidate geo_type_075 --seeds 0 --output predictions/e125_competitive_geo
```

候选的训练/生成步骤、输入文件校验值及配置见输出目录内的 manifest。代码开发、调试和结果汇总使用了 Codex 辅助，实验记录供组员复核。原始矩阵、评价子集和预测文件存放在授权的本地环境。
