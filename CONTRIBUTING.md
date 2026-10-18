# 小组协作说明

## 每次修改的基本流程

先完成 GitHub 登录并获得仓库访问权限。首次在自己的电脑提交前，为 Git 配置个人姓名和邮箱，便于区分四位成员的贡献。

在项目根目录开始一次修改：

```sh
git switch main
git pull --ff-only
git switch -c docs/revise-proposal
```

上面的分支名只是示例，每项工作使用不同且能说明用途的名称，例如 `feat/baseline`、`analysis/data-overview`。

修改并检查文件后，只提交本次需要的文件：

```sh
git status
git add t1_proposal_latex/main.tex 统计学习开题报告_T1_讨论稿.pdf
git commit -m "docs: revise T1 proposal"
git push -u origin HEAD
```

随后在 GitHub 创建 Pull Request，写清修改内容与检查方式，邀请队友审阅后合并到 `main`。若有冲突，先与正在修改同一部分的队友协调；不要强制推送覆盖他人的提交。

## 报告与实验约定

- 以 LaTeX 源码为编辑入口；提交报告改动时同时更新根目录 PDF，并检查中文、公式、页数和参考文献。编译方法见首页。
- 使用 GitHub Issues 记录待办、负责人和讨论结论。四人的具体分工由小组讨论后补充。
- 实验代码应附运行方法与依赖说明，记录数据版本、参数、随机种子及评价结果。未运行的方案和已验证的结果要清楚区分。
- 不上传原始竞赛数据、访问令牌、密码、本地环境文件、模型权重和大批预测结果。需要共享时，在文档中记录合规的获取方式与文件校验值。
- 使用 GPT 等工具辅助时，由提交者核实公式、引用、代码行为与结果，按课程及竞赛要求记录使用情况。
