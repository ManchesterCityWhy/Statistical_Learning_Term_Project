# T1 实验归档（2026-11-24）

`runs/` 收录八次已完成的本地开发实验。每次包含逐模型、逐种子的 `metrics.csv`，整体 `summary.csv`，以及记录配置、输入校验值、软件版本和代码指纹的 `manifest.json`。共 16 份 CSV、105 条评分记录。分数来自 E10.5 本地评价，不能作为 E12.5 官方测试成绩。

| 目录 | 内容 |
| --- | --- |
| `dev_20261124`、`matched_reference_20261124` | 早期七种方法的验证，以及使用相同参考细胞的复核 |
| `competitive_screen` | 保零、重加权和稀疏表达模型的第一次筛选 |
| `competitive_verify_0`、`competitive_verify_31`、`competitive_verify_101` | 固定参考细胞后的三轮内部方法比较 |
| `competitive_bulk`、`competitive_geo_transport` | 两条外部数据方法的三种子验证 |
| `prediction_manifests/` | 五份本地 E12.5 预测的生成配置与格式检查记录 |

`nonzero_fraction` 是模型输出的非零比例，`mmd_u` 和 `variogram` 分别对应 MMD 和 CSS 的原始值，数值越低越好。各轮用自己的参考与评价抽样校准。单次抽样的标准差留空。部分 manifest 的 `git_head` 标识当时的本地工作版本，脚本内容可用相应的 SHA-256 核对。

公开归档只包含模型及运行级的记录。原始表达矩阵、验证数据、细胞索引和预测矩阵存放在本地。公开基因清单可通过 `python -m t1_pipeline sync-specs` 从官方获取。[数据规则](https://virtualembryo.ai/challenge/rules) · [复现说明](../../README.md)
