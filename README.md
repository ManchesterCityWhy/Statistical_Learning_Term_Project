# 统计学习课程大作业 · Virtual Embryo Challenge T1

2026 秋季《统计学习》期末项目。我们用小鼠胚胎心脏区域的单细胞 RNA 数据预测未来阶段的基因表达分布。开发阶段用 E8.5、E9.5 预测 E10.5，竞赛决赛获准使用 E10.5 后，向官方提交了两份 E12.5 预测。E12.5 真值由主办方保留。

- **期末报告** [PDF](reports/final/T1_最终报告.pdf) · [LaTeX 源码](reports/final/T1_最终报告.tex)
- **实验记录** [第一至九轮索引与复现说明](experiments/README.md)
- **第二次提交复盘** [网页](reports/final/T1_第二次提交复盘.md) · [PDF](reports/final/T1_第二次提交复盘.pdf)

## 官方测试成绩

| 提交 | 方法 | 总分 | DES | DCS | MMD | CSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第一次 | 第二轮保守版 h1，基于 bulk 趋势的细胞组成调整 | 70.62 | 66.3 | 76.9 | 69.0 | 70.5 |
| 第二次 | 第九轮，结合 E14.5 参照调整组成和表达，并对源细胞做测序深度抽稀 | **73.37** | 65.8 | 82.3 | 72.4 | 73.0 |

第二次比第一次高 **2.75 分**，提交时在 554 支队伍中列第 352 名。两次正式提交机会均已使用。每项分数口径、榜单参照和第一次反馈如何影响后续选模见[期末报告](reports/final/T1_最终报告.pdf)和[复盘](reports/final/T1_第二次提交复盘.md)。

本地 E10.5 验证中，第二轮的最佳配置在选模集和复核集上分别得到 60.50、59.79 分，第三轮熵平衡方案分别得到 57.53、56.50 分。两者共享第二轮的抽样划分。本地分数对应不同的目标阶段，单独列于[实验记录](experiments/README.md)。

## 代码与复现

- [环境、数据和命令行](t1_pipeline/README.md) 包括依赖安装、官方基因清单同步、本地评价及预测格式检查。
- [实验索引](experiments/README.md) 按轮列出方法代码、评分表、运行记录和导出清单。
- [第一次提交清单](experiments/round2_seed42/prediction_h1_manifest.json)与[第二次提交清单](experiments/round9_seed42/manifest.json)记录配置、校验值及格式检查。清单中的 `not_submitted` 描述文件生成时的状态，正式成绩以上表和复盘为准。

安装运行依赖后，在仓库根目录执行 `python -m pytest -q` 可运行测试。三个报告图已随仓库保存。需要重绘时，在同一环境安装 Matplotlib 后运行 `python reports/final/make_figures.py`。

用 XeLaTeX 编译期末报告，确保保留仓库自带的中文字体。

```sh
cd reports/final
xelatex -interaction=nonstopmode -halt-on-error T1_最终报告.tex
xelatex -interaction=nonstopmode -halt-on-error T1_最终报告.tex
```

使用 Tectonic 也可在同一目录运行 `tectonic T1_最终报告.tex`。开题报告的编译步骤见[字体与编译说明](t1_proposal_latex/README.md)。

## 课程资料

- [课程大作业说明](课程大作业说明.txt)
- [开题报告 PDF](统计学习开题报告_T1_讨论稿.pdf) · [LaTeX 源码](t1_proposal_latex/main.tex)
- [中期报告公开版 PDF](reports/T1_中期报告_公开版.pdf) · [LaTeX 源码](reports/T1_中期报告_公开版.tex)
- [仓库协作说明](CONTRIBUTING.md)

## 数据来源与使用

竞赛数据由 UC San Diego 的 Qingquan Zhang 和 Neil Chi 团队生成，由 Virtual Embryo Challenge 提供。竞赛原始矩阵、逐细胞来源索引及预测矩阵保存在各自的授权环境中。下载与使用要求见[官方数据说明](https://virtualembryo.ai/challenge/data)和[竞赛规则](https://virtualembryo.ai/challenge/rules)。

中文字体 Noto Serif SC 采用 [SIL Open Font License 1.1](t1_proposal_latex/fonts/LICENSE.txt)。课程材料、竞赛数据和字体各自适用相应条款。
