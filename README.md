# 新股网下中签率预测项目

## 项目简介

本项目是一个面向 A 股新股申购场景的 AI 课题项目，目标是基于历史新股发行、初步询价、网下申购和配售数据，建立网下中签率预测模型，并最终形成一个可演示、可回测、可部署到网页的预测工具。

课题题目可暂定为：

```text
基于历史新股发行数据的网下中签率预测与申购决策辅助工具
```

核心思路是：在“初步询价开始前”这一正式预测时点，使用当时已经可获得的招股书、询价公告、发行安排、估值和历史市场环境数据预测新股的网下超额认购倍数，再反推出网下中签率。初步询价结果、最终发行价、回拨、网下申购、配售和上市后数据只能用于训练标签、事后解释或研究对照，不能进入正式预测输入。

```text
y = log(网下超额认购倍数)
预测网下中签率 = 1 / exp(y_pred)
```

## 业务背景

新股申购是 A 股市场的重要投资行为，中签率高低直接影响投资者的资金安排和申购策略。网下中签率受发行规模、网下配售数量、询价对象数量、申购总量、战略配售比例、市值门槛、板块制度和市场热度等因素影响。

传统判断方式通常依赖经验和事后统计，本项目希望通过系统整理 Wind 历史发行数据，建立可解释、可回测的预测模型，为新股申购决策提供量化参考。

## 预测目标

优先预测目标：

```text
log(网下超额认购倍数)
```

备选或辅助目标：

```text
网下投资者中签率
网下申购配售比例
网下申购总量
网下有效报价申购量
网下超额认购倍数(回拨前)
```

优先使用对数超额认购倍数的原因：

- 原始中签率通常很小，直接回归容易不稳定。
- 超额认购倍数和中签率存在清晰反向关系。
- 对超额认购倍数取对数可以缓解偏态分布。
- 预测结果容易转换成投资者能理解的中签率。

## 样本范围

本项目重点关注注册制之后的新股样本。

| 板块 | 初步样本口径 | 说明 |
| --- | --- | --- |
| 主板 | 2023 年全面注册制之后 | 需要切分注册制前后样本，第一阶段优先使用注册制后 |
| 创业板 | 2020 年注册制改革之后 | 需要切分注册制前后样本 |
| 科创板 | 开板以来 | 科创板一开始就是注册制，可作为完整注册制样本 |
| 北交所 | 单独建模或单独评估 | 流动性与参与者结构特殊，不宜直接假设与其他板块同质 |

## 数据字段分层

### 正式预测输入字段（询价前）

这些字段应尽量满足预测时点可获得原则：

```text
网下申购数量上限
网下申购数量下限
网下申购步长
网下配售数量
战略配售获配股份数
战略配售获配股份占比
主承销商战略获配股份数
主承销商战略获配股份占比
网下投资者分类限售配售方式
网下投资者分类配售限售比例
网下询价市值门槛
网下询价市值门槛(A类)
网下询价市值门槛(主题与战略)
发行价格下限(底价)
发行价格上限
行业市盈率、可比公司估值
近期已上市 IPO 首日表现、同板块破发率、市场成交额、市场涨跌幅
同期 IPO 批次竞争强度
```

### 研究对照字段

以下字段发生在询价后或更晚，禁止进入正式预测模型，只能用于研究对照、信息增益分析或事后解释：

```text
初步询价申报价格
网下申报价格加权平均数
网下申报价格中位数
初步询价申报数量
初步询价配售对象家数
初步询价询价对象家数
初步询价申购总量
初步询价申购倍数(回拨前)
最终发行价、发行市盈率、市净率、募集资金总额
剔除无效和最高报价后申购总量
剔除无效和最高报价后配售对象
剔除无效和最高报价后询价对象
回拨比例、回拨后网上/网下发行量
```

### 训练标签字段

这些字段主要用于构造监督学习标签，不能直接作为正式预测输入：

```text
网下申购配售比例
网下投资者中签率
网下超额认购倍数
网下超额认购倍数(回拨前)
网下申购总量
网下有效报价申购量
```

### 辅助参考字段

网上发行相关字段可用于市场热度分析，但很多属于事后数据，正式建模前需要逐项确认是否会造成数据泄露。

## 项目路线图

### 1. 取数据

目标是从 Wind 或其他数据源整理注册制后新股发行样本。

当前计划：

- 导出主板、创业板、科创板、北交所 IPO 样本。
- 按板块和注册制时间切分样本。
- 建立原始字段字典。
- 标记字段发布时间和预测时点可用性。
- 保存原始数据、清洗数据和建模数据三个层级。

### 2. 分析数据

需要完成：

- 样本数量统计。
- 字段缺失率统计。
- 标签分布分析。
- 分板块对比分析。
- 异常值检查。
- 特征与标签相关性分析。
- 注册制前后样本差异分析。

重点观察：

- 网下超额认购倍数是否随板块显著不同。
- 初步询价申购总量、询价对象家数、配售对象家数与中签率之间的关系。
- 网下配售数量和战略配售比例对中签率的影响。
- 北交所是否需要完全独立建模。

### 3. 建立多种预测模型

第一阶段模型：

```text
历史均值模型
滚动均值模型
板块均值模型
线性回归
Ridge / Lasso
Random Forest
XGBoost / LightGBM / CatBoost
```

第二阶段模型：

```text
全市场模型 + 板块特征
板块专属模型
全市场模型 + 板块残差校准
相似新股检索 + 机器学习预测融合
```

### 4. 搭建回测系统

回测系统需要模拟真实预测过程，避免未来函数。

基本要求：

- 按发行日期排序。
- 使用过去样本训练，预测未来样本。
- 支持滚动窗口和扩展窗口。
- 分板块输出误差指标。
- 比较不同模型的预测能力。
- 支持策略层面评估，例如优先申购预测中签率较高的新股。

建议评价指标：

```text
MAE
RMSE
MAPE
R2
Spearman 排名相关
分板块误差
分年度误差
高/中/低中签率分档准确率
```

### 5. 部署到网页

最终网页工具应包括：

- 新股信息输入区。
- 预测结果展示区。
- 历史相似新股展示区。
- 特征贡献或影响因素解释区。
- 模型版本和样本范围说明。

核心输出：

```text
预测网下超额认购倍数
预测网下中签率
风险分档
历史分位数
主要影响因素
```

## 板块建模问题

当前待研究的关键问题是：主板、创业板、科创板、北交所应该使用一个统一模型，还是分板块建模。

### 可选方案

| 方案 | 描述 | 优点 | 风险 |
| --- | --- | --- | --- |
| 全市场大模型 | 所有板块合并训练，加入板块特征 | 样本最多，第一版实现简单 | 可能忽略板块差异 |
| 分板块模型 | 每个板块单独训练模型 | 解释清晰，尊重制度差异 | 部分板块样本少 |
| 全市场模型 + 板块校准 | 先训练统一模型，再按板块修正预测残差 | 兼顾样本量和差异 | 实现和验证更复杂 |
| 分层模型 | 共享全市场信息，同时学习板块差异 | 理论上更稳健 | 对实现和样本量要求更高 |

### 初步建议

第一阶段不要过早押注单一方案。建议同时实现三条线：

```text
统一模型
统一模型 + 板块特征
板块专属模型
```

通过时间序列回测比较后，再决定最终生产模型。如果北交所误差显著偏高，应将北交所独立建模。

## 预期成果

项目最终应形成：

- 一份 Word 课题报告。
- 一份 PPT 答辩展示。
- 一套数据处理流程。
- 一套模型训练流程。
- 一套时间序列回测系统。
- 一个可演示的网页预测工具。
- 一份 AI 交互记录和开发过程说明。

## 当前待办

- [x] 确认 Wind 数据导出字段和文件格式。
- [x] 建立字段字典。
- [x] 建立注册制后样本筛选规则。
- [x] 完成主板、创业板、科创板、北交所样本切分。
- [x] 完成第一版 EDA。
- [x] 构造 `log(网下超额认购倍数)` 标签。
- [x] 建立三阶段（T-6 / T-1 / T+1）预测模型与时间序列回测。
- [x] 比较统一模型与板块专属模型。
- [x] 部署 Streamlit 网页演示。

## 2026-05-21 初步 EDA 记录

本次处理了 3 个 Wind 导出文件：

```text
D:/wind导出数据/全部科创板_网下打新数据.xlsx
D:/wind导出数据/注册制创业板_网下打新数据.xlsx
D:/wind导出数据/全部科创板_主板.xlsx
```

处理脚本：

```text
scripts/initial_data_analysis.py
```

主要输出：

```text
data/processed/ipo_offline_sample.csv
data/processed/ipo_offline.db
outputs/initial_analysis/initial_analysis_report.md
outputs/initial_analysis/field_dictionary.csv
outputs/initial_analysis/missing_required_prediction_fields.csv
outputs/initial_analysis/missing_by_field.csv
outputs/initial_analysis/descriptive_stats.csv
outputs/initial_analysis/board_summary.csv
outputs/initial_analysis/year_board_counts.csv
outputs/initial_analysis/correlation_pre_subscription_like.csv
outputs/initial_analysis/correlation_all_fields.csv
outputs/initial_analysis/outlier_checks.csv
outputs/initial_analysis/figures/label_distribution_by_board.svg
outputs/initial_analysis/figures/yearly_median_oversubscription.svg
outputs/initial_analysis/figures/top_missing_fields.svg
outputs/initial_analysis/figures/top_pre_subscription_correlations.svg
```

样本概况：

- 合计 1,326 条 IPO 样本。
- 科创板 611 条，创业板 603 条，主板注册制后 112 条。
- 有 `网下超额认购倍数` 标签的样本 1,194 条。
- 上市日期范围为 2019-07-22 至 2026-05-18。
- 当前尚未纳入北交所。

初步结论：

- SQLite 适合作为清洗后数据中间层和回测查询层，当前已生成 `ipo_offline_sample` 表；建模和复杂特征工程仍建议使用 Python。
- 板块差异明显：主板注册制后网下超额认购倍数中位数约 8,622.80 倍，创业板约 3,367.41 倍，科创板约 2,596.93 倍，后续必须分板块评估。
- `网下申购配售比例` 与 `100 / 网下超额认购倍数` 基本互为倒数，只能作为标签或校验字段，不能作为正式预测输入。
- 当前 Excel 中很多强相关字段属于申购后或配售后信息，例如网下申购总量、配售对象家数、配售比例，后续建模时要严格排除泄露。
- 下一步最关键的数据缺口是初步询价阶段可见字段，包括申购上限/下限/步长、初步询价申报数量、询价对象家数、市值门槛、剔除最高报价后对象数量等。
- [ ] 建立第一版基准模型。
- [ ] 建立树模型并输出特征重要性。
- [ ] 搭建时间序列回测框架。
- [ ] 比较统一模型、板块模型和板块校准模型。
- [ ] 设计网页预测工具原型。

## 2026-05-22 开发进展：三阶段模型 + 网页演示已上线

第一轮端到端流程（取数 → 分析 → 建模 → 回测 → 部署）已全部跑通。

2026-05-27 与领导确认后，正式预测节点从“询价完成后、申购前”调整为“询价开始前”。因此当前网页和 CLI 默认使用 T-6 模型；T-1/T+1 只保留为研究对照。

### 三阶段时点框架

为彻底杜绝未来数据泄露，所有特征按“信息释放时点”打标签，并据此构造三个模型：

| 阶段 | 时点 | 可用信息 | 用途 |
| --- | --- | --- | --- |
| **T-6** | **询价前正式预测** | 招股书、询价公告、行业 PE、历史市场热度、市场流动性/情绪、批次竞争 | **当前网页默认模型** |
| T-1 | 询价后研究对照 | T-6 全部 + 询价结果（询价超额认购倍数、机构家数、报价分布、最终发行价） | 信息增益分析 / 历史对照 |
| T+1 | 回拨后研究对照 | T-1 全部 + 回拨比例 | 事后校验 / 上界参考 |
| T+2 | — | 网下超额认购倍数（目标变量） | 永不作为输入 |

网页默认使用 **T-6 LightGBM** 作为正式预测口径：询价开始前即可运行，不使用询价结果、网下申购、配售、回拨或上市后数据。T-1 / T+1 保留为研究对照模型，用于展示信息逐步释放后的预测上界，不作为当前正式预测口径。

### 回测结果（扩张窗口 OOS，按申购截止日排序）

| 模型 | 阶段 | n | MAE | R² | Spearman |
| --- | --- | --- | --- | --- | --- |
| board_mean_t6 | T-6 | 624 | 0.370 | 0.529 | 0.270 |
| lgbm_t6 | T-6 | 624 | 0.307 | 0.666 | 0.619 |
| ridge_t1 | T-1 | 624 | 0.388 | -0.241 | 0.671 |
| lgbm_t1（研究对照） | T-1 | 624 | 0.109 | 0.913 | 0.961 |
| lgbm_t1plus | T+1 | 624 | 0.102 | 0.920 | 0.965 |

- 正式模型（T-6）OOS Spearman 为 0.619；询价结果释放后的 T-1/T+1 仍只作为研究对照，不作为正式预测口径。
- 分板块 OOS Spearman（T-1）：科创板 0.986、创业板 0.979、北交所 0.886（n=6）、主板 0.640。
- 板块专属模型未优于统一模型（主板 -0.23、创业板 -0.02、科创板 +0.001），统一模型 + 板块特征 + 跨板块学习为当前最优方案。
- 已补充 4 个 T-6 市场环境特征（见下）。`market_turnover_ma20`（沪深两市近 20 日日均成交额）在 lgbm_t6 中重要性排名第 2；T-6 整体 Spearman 0.488→0.512、T-1 研究对照 0.951→0.953、T+1 0.967→0.970，均小幅提升。
- 2026-05-29 新增并纳入确定口径 T-6 因子：网下询价市值门槛、预计募资额、近一年营收、三年营收 CAGR、板块滚动行情、主承销商历史表现、申万一级行业代码历史 IPO 热度。暂未纳入 `issue_pb_factor`、发行价格区间、行业行情滚动因子等口径仍需确认或缺映射的数据。

### 运行方式

```bash
# 0.（可选，一次性）市场流动性/情绪特征数据
#    从 Wind 导出万得全A(881001.WI) 日成交额 amt + 涨跌幅 pct_chg，转换为 market_daily.csv
python scripts/convert_wind_market.py --in "D:/wind导出数据/全A成交额及涨跌幅数据.xlsx"

# 1. 数据清洗与 EDA（生成 SQLite 与分析产出）
python scripts/initial_data_analysis.py

# 2. 三阶段建模 + 回测 + 保存模型
python scripts/baseline_models.py

# 3. 板块专项模型对比（可选）
python scripts/board_models.py

# 4. 询价前因子洞察与板块/时期特征报告
python scripts/factor_insights.py

# 5. 命令行预测
python scripts/predict.py --code 688041 --stage T6

# 6. 启动网页演示
streamlit run app.py
```

### 代码与产出物

```text
scripts/convert_wind_market.py     Wind 市场日度导出 → market_daily.csv（市场流动性特征）
scripts/initial_data_analysis.py   数据清洗 + EDA
scripts/baseline_models.py         三阶段建模 + 时间序列回测
scripts/board_models.py            板块专项模型对比
scripts/factor_insights.py         询价前因子洞察 + 板块/时期特征报告
scripts/predict.py                 模型加载与预测（CLI / API / 板块路由）
app.py                             Streamlit 网页演示
data/processed/ipo_offline.db      清洗后 SQLite（回测/查询层）
outputs/initial_analysis/          EDA 报告、字段字典、相关性、缺失率
outputs/baseline_models/           回测指标、特征重要性、序列化模型、报告
outputs/board_models/              板块对比指标与报告
outputs/factor_insights/           因子IC、分组、SHAP贡献、板块时期画像和领导速览报告
项目汇报_新股网下中签率预测.md      面向领导的汇报文档
需要完善的部分.md                  后续开发 backlog（待完善增强项）
```

> 后续待完善项（市场流动性特征、SHAP 解释、增量数据入库、按股票名称预测）详见 `需要完善的部分.md`。

### 已知限制

- 主板注册制后有标签样本仅约 105 条，受小样本上限制约询价前 OOS 偏弱，进一步提升空间有限，暂不作为优先项。
- 北交所有标签样本仅 41 条且受机制限制难再扩充，暂与其他板块合并训练、单独评估，未独立建模。
- 网页对“已入库历史股票”的查询为样本内结果（偏乐观）；真实无泄漏成绩见回测产出 `outputs/baseline_models/predictions.csv`。

## GitHub 协作与部署

### 仓库内容边界

建议提交到 GitHub 的核心内容：

- `app.py`：Streamlit 网页演示入口。
- `scripts/`：数据清洗、市场数据转换、建模、回测、预测 API/CLI。
- `data/processed/`：可直接复现 demo 的轻量处理后数据。
- `outputs/initial_analysis/`：字段字典、EDA 报告和关键图表。
- `outputs/baseline_models/`：三阶段模型、回测指标、特征重要性和预测明细。
- `outputs/board_models/`：板块模型对比产出。
- `README.md`、`AGENTS.md`、`需要完善的部分.md`：项目说明、协作规则和 backlog。
- `requirements.txt`：Python 依赖。

不建议提交：

- Wind/Tushare 原始导出 Excel、个人下载路径和未脱敏数据。
- `.venv/`、`__pycache__/`、`.claude/`、编辑器配置和本地日志。
- `outputs/peek_*.json` 等临时探查文件。
- `reveal/`、`*_slides.html`、一次性 HTML 构建产物。

### 新机器启动

```bash
git clone <你的GitHub仓库地址>
cd <仓库目录>

python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt

streamlit run app.py
```

命令行预测：

```bash
python scripts/predict.py --code 688041
python scripts/predict.py --code 688041 --stage T6
```

重新生成数据和模型：

```bash
python scripts/initial_data_analysis.py
python scripts/baseline_models.py
python scripts/board_models.py
```

### 推送到 GitHub

第一次推送时：

```bash
git add .gitignore .gitattributes README.md AGENTS.md requirements.txt app.py scripts data/README.md data/processed outputs/README.md outputs/initial_analysis outputs/baseline_models outputs/board_models 需要完善的部分.md
git commit -m "Prepare IPO predictor repo for collaboration"
git branch -M main
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

后续协作：

```bash
git pull --rebase
git checkout -b codex/<任务名>
# 修改代码、数据或报告
git add <相关文件>
git commit -m "<简短说明>"
git push -u origin codex/<任务名>
```

重要规则：任何新增字段、特征、模型或回测规则，都要同步更新 `README.md` / `AGENTS.md`，并说明是否属于预测前可用变量，避免未来数据泄露。

## 文档维护规则

随着项目推进，需要持续更新：

- `README.md`：记录课题目标、数据口径、技术路线、实验结果和部署说明。
- `AGENTS.md`：记录开发协作规则、字段使用约束、建模注意事项和后续 Agent 工作规范。

任何重要设计变化都应写入文档，包括：

- 样本范围变化。
- 标签定义变化。
- 特征集合变化。
- 模型选择变化。
- 回测方式变化。
- 部署方案变化。
