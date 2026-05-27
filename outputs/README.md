# 产出目录说明

本仓库保留关键实验产出，目的是让协作者克隆后可以直接查看指标、加载模型并运行网页演示。

## 建议纳入 Git

- `outputs/initial_analysis/`：字段字典、缺失率、描述统计和 EDA 图表。
- `outputs/baseline_models/`：三阶段模型的回测指标、特征重要性、预测明细和序列化模型。
- `outputs/board_models/`：板块模型对比指标、报告和模型文件。

## 默认忽略

- `outputs/peek_*.json`：临时探查文件。
- 临时日志、一次性截图、手工导出的 HTML 构建产物。

如果模型文件未来超过 GitHub 普通仓库的舒适范围，建议迁移到 Git LFS、Release 附件或对象存储，并在 `outputs/baseline_models/manifest.json` 中记录模型版本和下载地址。
