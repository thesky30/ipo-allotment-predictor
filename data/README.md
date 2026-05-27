# 数据目录说明

本仓库默认提交 `data/processed/` 下的轻量处理后数据，便于多端和多人直接复现网页演示与回测查询。

## 已纳入 Git 的数据

- `data/processed/ipo_offline_sample.csv`：清洗后的 IPO 样本表，供建模与人工检查使用。
- `data/processed/ipo_offline.db`：由清洗流程生成的 SQLite 查询层，网页和 CLI 会读取该文件。
- `data/processed/market_daily.csv`：市场流动性/情绪特征的日频中间表。

## 不纳入 Git 的数据

- `data/raw/`：Wind、Tushare 或其他来源的原始导出文件。
- `data/external/`：第三方临时下载或人工补充文件。
- `*.xlsx` / `*.xls`：本地原始 Excel 导出默认忽略。

如果后续原始数据需要团队共享，优先放到团队网盘或对象存储，并在 README 中记录下载位置、字段口径和生成日期。
