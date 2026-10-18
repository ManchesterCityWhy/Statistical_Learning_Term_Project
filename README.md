# Statistical Learning Term Project

2026 秋季《统计学习》课程大作业，四人小组共同维护。

选题为 Virtual Embryo Challenge **T1 Temporal**：利用较早发育阶段的单细胞 RNA 数据，预测未来阶段的细胞表达分布。目前已完成官方平台团队注册，并确认可下载 E8.5、E9.5 训练数据；正在完善开题和实验方案，具体研究设计以开题报告为准。

## 现有资料

- [开题报告 PDF](统计学习开题报告_T1_讨论稿.pdf)
- [开题报告 LaTeX 源码](t1_proposal_latex/main.tex)
- [课程大作业说明](课程大作业说明.txt)：来自课程说明原文，供组内查阅。
- [字体与编译说明](t1_proposal_latex/README.md)

```text
.
├── README.md
├── CONTRIBUTING.md
├── 课程大作业说明.txt
├── 统计学习开题报告_T1_讨论稿.pdf
└── t1_proposal_latex/
    ├── main.tex
    ├── README.md
    └── fonts/                   # 中文字体、许可与校验值
```

## 编译开题报告

完整保留 `t1_proposal_latex/fonts/`，使用 XeLaTeX 或 Tectonic。请从源码目录运行，保证相对字体路径正确。

使用 Tectonic：

```sh
cd t1_proposal_latex
mkdir -p ../build
tectonic --outdir ../build main.tex
```

或者使用已安装中文支持的 TeX Live / MacTeX，连续编译两次：

```sh
cd t1_proposal_latex
mkdir -p ../build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
```

产物为 `build/main.pdf`。检查中文、公式和分页后，在项目根目录运行以下命令，更新供队友直接阅读的 PDF：

```sh
cp build/main.pdf 统计学习开题报告_T1_讨论稿.pdf
```

`build/` 和编译缓存不纳入版本管理，根目录的报告 PDF 纳入版本管理。现有 PDF 文件名保留以维持原有链接，内容以最新源码编译结果为准。随附字体用于解决中文缺字和阅读器兼容问题，无需另行安装。

报告比较官方基线、全基因空间收缩外推与 PCA 收缩外推，并对群体表达和比例调整进行消融。E10.5 真值公开后用于开发验证及有限模型选择；方法与超参数固定后，使用 E9.5、E10.5 重新拟合并预测隐藏的 E12.5。

## 小组协作

仓库当前为公开仓库，任何人可读取和克隆；直接推送需要仓库写入权限，也可通过 Fork 和 Pull Request 贡献修改：

```sh
git clone https://github.com/ManchesterCityWhy/Statistical_Learning_Term_Project.git
cd Statistical_Learning_Term_Project
```

`main` 保存小组确认的版本；每次修改从最新 `main` 创建分支，通过 Pull Request 讨论和合并。源码调整后同步更新 PDF，避免两人同时直接编辑二进制 PDF。具体步骤见 [协作说明](CONTRIBUTING.md)。

近期工作：复核报告中的四人初步分工、完成训练数据校验，并逐步补充实验代码、环境说明和开发验证记录。

## 数据与实验

官方入口：[Virtual Embryo Challenge](https://virtualembryo.ai/challenge)，[数据说明](https://virtualembryo.ai/challenge/data)。

当前仓库不含竞赛数据。后续将原始数据、下载文件、模型权重和预测结果保存在本地，相关目录已加入忽略规则。实验代码应注明数据版本、预处理方法、随机种子、运行方式和评价结果；数据获取和使用遵守官方规则。

## 资料许可

随附 Noto Serif SC 字体采用 SIL Open Font License 1.1，完整许可见 [fonts/LICENSE.txt](t1_proposal_latex/fonts/LICENSE.txt)，来源见 [字体说明](t1_proposal_latex/README.md)。课程材料、竞赛数据与字体分别遵循其适用的使用规定；仓库暂未指定项目整体的开源许可。
