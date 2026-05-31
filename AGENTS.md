# AGENTS.md

## 项目定位

本项目围绕 A 股新股网下中签率预测展开，目标是建立一套可迭代的数据、建模、回测与网页部署流程。核心预测对象暂定为：

```text
y = log(网下超额认购倍数)
预测网下中签率 = 1 / exp(y_pred)
```

所有后续 Agent 或开发者在推进项目时，应优先保证预测时点一致：领导已于 2026-05-27 确认，正式预测节点提前到“初步询价开始前”。正式预测输入只能使用询价前已经可获得的数据，不能使用初步询价结果、最终发行价、回拨、网下申购、配售或上市后数据。

## 协作原则

- 项目文档需要随开发持续更新，尤其是 `README.md` 和本文件。
- 每次新增数据口径、特征、模型、回测规则或部署方案，都应同步记录设计理由。
- 对字段按“预测前可用变量”“训练标签”“事后解释变量”分类，避免数据泄露。
- 主板、创业板、科创板、北交所应保留板块差异，不要默认混为一个完全同质样本。
- 所有实验结果应记录样本范围、训练时间窗口、验证方式、评价指标和主要结论。

## 当前总体路线

1. 取数据
   - 主板：重点使用 2023 年全面注册制之后样本。
   - 创业板：重点使用 2020 年注册制改革之后样本。
   - 科创板：自开板以来即为注册制，可作为完整注册制样本。
   - 北交所：流动性、投资者结构、发行机制和参与者行为特殊，建议单独建模或至少单独评估。

2. 分析数据
   - 统计各板块样本数量、字段缺失率、异常值和时间分布。
   - 分析网下超额认购倍数、网下中签率、配售比例等标签分布。
   - 对比不同板块的询价对象数量、配售对象数量、网下配售规模、市值门槛、战略配售比例等结构差异。

3. 建立多种形态预测模型
   - 基准模型：历史均值、滚动均值、板块均值、相似新股均值。
   - 传统模型：线性回归、Ridge、Lasso。
   - 树模型：Random Forest、XGBoost、LightGBM、CatBoost。
   - 分层模型：全市场基础模型 + 板块特征，或全市场模型 + 板块专属微调。
   - 板块专属模型：主板、创业板、科创板、北交所分别训练和验证。

4. 搭建回测系统
   - 按真实时间顺序滚动训练和预测，避免未来数据泄露。
   - 分模型、分板块、分年份评估预测误差。
   - 比较模型在投资决策场景中的表现，例如预测中签率分位、申购优先级排序、策略收益模拟。

5. 部署到网页
   - 建立输入表单，正式预测只允许输入询价前可获得的新股发行、申购规则、估值和历史市场环境字段。
   - 输出预测网下超额认购倍数、预测网下中签率、置信区间或风险分档。
   - 展示关键影响因素、历史相似样本和模型版本信息。

## 板块建模策略（已得出结论）

> 以下三方案均已实现并经扩张窗口回测对比，**结论：统一模型 + 板块特征 ≥ 板块专属模型**（详见本节末「实际结论」）。原始候选方案保留如下供追溯。

当前需要重点比较以下方案：

### 方案 A：全市场大模型

将主板、创业板、科创板、北交所样本合并训练，在特征中加入板块类别。

优点：
- 样本量最大。
- 能学习跨板块共性规律。
- 适合作为第一版基线。

风险：
- 北交所和其他板块差异较大，可能拉低整体泛化效果。
- 主板注册制样本时间较短，可能被其他板块规律主导。

### 方案 B：全市场模型 + 板块微调

先训练一个全市场基础模型，再针对不同板块进行局部校准或微调。

可选做法：
- 对模型预测结果按板块做残差校准。
- 在不同板块上训练二阶段修正模型。
- 使用全市场模型做特征表达，再训练板块专属后处理模型。

优点：
- 兼顾样本量和板块差异。
- 适合主板注册制样本较少的情况。

风险：
- 实现复杂度高于单模型。
- 需要严格回测验证是否真的优于简单方案。

### 方案 C：板块专属模型

对主板、创业板、科创板、北交所分别建模。

优点：
- 最大程度保留板块制度、流动性和参与者结构差异。
- 模型解释更清晰。

风险：
- 主板注册制后样本可能偏少。
- 北交所样本可能存在流动性和数据口径问题，需要单独处理。

### 实际结论

三条线均已实现并经扩张窗口回测对比：

```text
全市场大模型
全市场模型 + 板块特征          ← 最终生产方案
主板/创业板/科创板/北交所板块专属模型
```

- **统一模型 + 板块特征 + 跨板块学习 ≥ 板块专属模型**：主板样本少（约 105 条）单独建模反而更差，不要默认"分板块就更好"。
- 北交所有标签样本仅约 41 条、机制特殊，**未独立建模**，改为与其他板块合并训练、单独评估。
- 方案 B 的残差校准 / 二阶段微调暂未实现，列入 backlog（见 `需要完善的部分.md`）。
- 最终生产口径：统一 **T-6 LightGBM**（含 `board` 板块特征 + 跨板块学习）。

## 字段使用规则

### 优先预测输入（询价前正式口径）

- 网下申购数量上限
- 网下申购数量下限
- 网下申购步长
- 网下配售数量
- 战略配售获配股份数
- 战略配售获配股份占比
- 主承销商战略获配股份数
- 主承销商战略获配股份占比
- 网下投资者分类限售配售方式
- 网下投资者分类配售限售比例
- 网下询价市值门槛
- 网下询价市值门槛(A类)
- 网下询价市值门槛(主题与战略)
- 发行价格下限(底价)
- 发行价格上限
- 行业市盈率、可比公司估值
- 近期已上市 IPO 首日表现、同板块破发率、市场成交额、市场涨跌幅
- 同期 IPO 批次竞争强度

### 研究对照字段（禁止进入正式预测）

- 初步询价申报价格
- 网下申报价格加权平均数
- 网下申报价格中位数
- 初步询价申报数量
- 初步询价配售对象家数
- 初步询价询价对象家数
- 初步询价申购总量
- 初步询价申购倍数(回拨前)
- 最终发行价、发行市盈率、市净率、募集资金总额
- 剔除无效和最高报价后申购总量
- 剔除无效和最高报价后配售对象
- 剔除无效和最高报价后询价对象
- 回拨比例、回拨后网上/网下发行量

### 推荐训练标签

- 网下超额认购倍数
- 网下超额认购倍数(回拨前)
- 网下申购配售比例
- 网下投资者中签率
- 网下申购总量
- 网下有效报价申购量

优先标签：

```text
y = log(网下超额认购倍数)
```

### 工程因子字典（实际字段名 → 阶段）

以下为代码中真实使用的字段名，时点归属由 `scripts/baseline_models.py` 的 `FEATURE_NODES` 唯一定义。**完整的来源与预期方向见 `README.md` 的「因子字典（特征全集）」**；此处供改字段时快速核对阶段，判断错误 = 数据泄露。

**✅ T-6 询价前（正式生产口径，约 42 个，`lgbm_t6` 唯一可用）**

- 板块结构：`board`
- 发行结构：`total_issue_shares_10k`、`offline_issue_before_clawback_10k`、`online_issue_before_clawback_10k`、`offline_issue_before_share_pct`
- 战略配售：`strategic_allocation_10k`、`strategic_allocation_share_pct`
- 申购规则：`subscription_upper_limit_10k`、`subscription_lower_limit_10k`、`subscription_step_10k`、`offline_market_value_threshold_10k_yuan`
- 发行定价（科创板）：`offer_price_upper_yuan`、`offer_price_lower_yuan`、`offer_price_range_pct`
- 发行/公司规模与成长：`expected_fundraising_100m_yuan`、`log_expected_fundraising`、`latest_revenue_100m_yuan`、`log_latest_revenue`、`revenue_cagr_3y_pct`
- 估值：`industry_pe_at_ipo`、`comparable_pe_avg_ex_nonrecurring`
- 市场流动性/情绪：`market_turnover_ma20`、`market_turnover_pct_rank_1y`、`market_turnover_ma20_over_ma60`、`market_return_ma20`
- 市场热度：`recent_ipo_first_day_return_ma20`、`same_board_break_rate_ma10`
- 批次竞争：`concurrent_ipo_count`、`same_board_concurrent_ipo_count`、`concurrent_offline_issue_sum_10k`
- 板块流动性/情绪：`board_turnover_ma20`、`board_turnover_pct_rank_1y`、`board_turnover_ma20_over_ma60`、`board_return_ma20`
- 承销商声誉：`underwriter_prior_ipo_count`、`underwriter_prior_log_oversub_mean`、`underwriter_prior_first_day_return_mean`、`underwriter_prior_break_rate`
- 行业历史热度：`sw_l1_prior_ipo_count`、`sw_l1_prior_log_oversub_mean`、`sw_l1_prior_first_day_return_mean`、`sw_l1_prior_break_rate`

**⚠️ T-1 询价后（仅研究对照，禁止进正式模型）**

`offer_price_yuan`、`offer_price_position_in_range`、`ipo_pe_diluted`、`issue_pb`、`pe_vs_industry`、`pe_vs_comparable`、`issue_amount_100m_yuan`、`inquiry_subscription_total_10k`、`inquiry_investors_count`、`inquiry_allotment_accounts`、`inquiry_oversubscription_ratio`、`quote_price_weighted_avg`、`quote_price_median`、`quote_price_vs_offer`、`excluded_subscription_share_pct`、`high_price_excluded_subscription_share_pct`

**⚠️ T+1 回拨后（仅研究对照）**

`clawback_ratio_pct`、`offline_issue_final_10k`、`offline_issue_final_share_pct`、`online_issue_final_10k`

**🚫 T+2 申购/配售结果（目标变量本身，永不作为输入）**

`offline_oversubscription_ratio`（及对数 `log_offline_oversubscription` = 标签）、`offline_allotment_ratio_pct`、`offline_subscription_total_10k`、`offline_valid_quote_subscription_10k`、`offline_allotment_accounts`、`offline_inquiry_investors`、`a_investor_*`、`offline_oversubscription_ratio_before_clawback`

**已知暂未纳入**：`issue_pb_factor`（可能依赖最终发行价）、发行价格区间（非科创板数据全空）、行业行情滚动因子（缺申万代码-名称映射）。

## 代码与实验记录要求

- 数据处理脚本应记录输入数据来源、字段映射、过滤条件和输出文件。
- 模型训练脚本应记录模型名称、特征集合、训练窗口、验证窗口和随机种子。
- 回测结果应保存为结构化文件，便于后续生成图表和报告。
- 网页部署前应固定一个可复现实验版本，记录模型文件、特征列表和训练样本范围。
- **部署 Python 版本必须固定为 3.11**（与 devcontainer 和保存 `.joblib` 的环境一致）。Streamlit Community Cloud 默认拉最新 Python（如 3.14），该 ML 栈无 3.14 wheel，源码编译后环境残缺会报 `ModuleNotFoundError: joblib`。仓库已含 `.python-version=3.11`，并须在 Streamlit Cloud 后台 Settings 把 Python version 设为 3.11（权威设置）后 Reboot。

## 当前开发进展（截至 2026-05-22）

整体五步路线已全部跑通一轮，从取数到网页演示均已落地，进入“可演示 + 可迭代”阶段。

### 已完成

- [x] 明确 Wind 字段导出方式和原始数据文件格式（4 份网下打新 Excel + 3 份询价补充）。
- [x] 建立字段字典和字段分层表（`outputs/initial_analysis/field_dictionary.csv`、`outputs/baseline_models/feature_time_nodes.csv`）。
- [x] 确定注册制样本过滤规则并完成板块划分（主板 2023+、创业板 2020+、科创板全量、北交所单列）。
- [x] 完成第一版数据质量分析（`scripts/initial_data_analysis.py` → `outputs/initial_analysis/`）。
- [x] 建立基准模型（板块均值 `board_mean_t6`）。
- [x] 建立树模型与特征重要性分析（LightGBM 三阶段 + Ridge 对照）。
- [x] 建立时间序列回测框架（扩张窗口，按申购截止日排序，切分点 2022/2023/2024/2025）。
- [x] 比较统一模型与板块专属模型（`scripts/board_models.py`，结论：统一模型≥板块专属）。
- [x] 设计并实现网页输入输出交互（`app.py`，Streamlit，三种输入方式 + 三阶段切换）。

### 核心架构：三阶段时点框架

所有特征按信息释放时点打标签（`FEATURE_NODES`，见 `scripts/baseline_models.py`），严格隔离未来数据：

| 阶段 | 时点 | 可用信息 | 模型 |
|---|---|---|---|
| **T-6** | **询价前正式预测** | 招股书 / 询价公告 / 行业 PE / 历史热度 / 市场环境 | **`lgbm_t6`、`board_mean_t6`** |
| T-1 | 询价后研究对照 | T-6 全部 + 询价结果（询价超额认购、机构数、价格分布、最终发行价） | `lgbm_t1` |
| T+1 | 回拨后研究对照 | T-1 全部 + 回拨比例 | `lgbm_t1plus` |
| T+2 | 目标变量 | 网下超额认购倍数 | 永不作为输入 |

正式网页和 CLI 默认模型固定为 **T-6 LightGBM**，不使用任何询价结果、回拨、网下申购、配售或上市后数据。T-1/T+1 只保留为研究对照，用于展示信息增益和模型上界。

### 关键结论（供后续 Agent 参考，避免重复踩坑）

- 领导已确认核心预测节点是询价之前，因此 **T-6 才是正式生产口径**；T-1/T+1 虽然精度更高，但含询价或回拨后信息，不能作为正式模型。
- 询价结果是预测价值的主要来源：OOS Spearman 从 T-6 的 0.49 跃升到 T-1 的 0.95；这只能作为信息增益分析，不能误写成正式效果。
- LightGBM 显著优于 Ridge（T-1 OOS Spearman 0.95 vs 0.74），关系高度非线性。
- 统一模型（含 board 特征 + 跨板块学习）≥ 板块专属模型；主板因样本少（约 105）单独建模反而更差，**不要默认分板块就更好**。
- `recent_ipo_first_day_return_ma20` 已按“过去 20 只已上市 IPO 首日涨幅”滚动计算，向后看、无泄漏。
- 已补充 4 个 T-6 市场环境特征（2026-05-25）：`market_turnover_ma20`（沪深两市/全A 近20日日均成交额，来源 Wind 万得全A，经 `scripts/convert_wind_market.py` 转换）、`market_return_ma20`（近20日涨跌幅）、`concurrent_ipo_count`（申购截止日 ±7 天批次竞争）、`same_board_break_rate_ma10`（同板块已上市近10只破发率）。全部严格向后看；`market_turnover_ma20` 在 lgbm_t6 中重要性排名第 2。OOS：T-6 0.488→0.512。市场两列对 T-6 的增益有时间区制敏感性，整体净正向。
- 已纳入一批确定口径 T-6 因子（2026-05-29）：网下询价市值门槛、预计募资额、近一年营收、三年营收 CAGR、板块滚动行情、主承销商历史表现、申万一级行业代码历史 IPO 热度。`issue_pb_factor`、发行价格区间、行业行情滚动因子暂不入模，分别因可能依赖最终价、当前全空、缺少申万代码-名称映射。重跑后正式 `lgbm_t6` OOS Spearman=0.619。
- 保存的全量模型对“已入库历史股票”的逐股查询属样本内（偏乐观）；某只股票的真实无泄漏成绩须查 `outputs/baseline_models/predictions.csv`（回测产出）。
- 文档同步（2026-05-31）：完整因子字典（T-6/T-1/T+1/T+2 全集 + 来源 + 预期方向）已写入 `README.md` 的「因子字典（特征全集）」；README 各章节已由「规划口吻」更新为「已完成」口径；本文件「字段使用规则」补充了实际字段名 → 阶段的工程因子字典。改字段时以 `FEATURE_NODES` 为唯一真源，并同步这两份文档。

### 代码与产出物地图

```text
scripts/convert_wind_market.py     Wind 市场日度导出 → data/processed/market_daily.csv
scripts/fetch_market_data.py       （备用）Tushare 拉成交额/指数 → data/processed/market_daily.csv
scripts/initial_data_analysis.py   数据清洗 + EDA  → data/processed/ + outputs/initial_analysis/
scripts/baseline_models.py         三阶段建模 + 回测 → outputs/baseline_models/
scripts/board_models.py            板块专项模型对比 → outputs/board_models/
scripts/factor_insights.py         询价前因子洞察 + 板块/时期画像 → outputs/factor_insights/
scripts/model_classes.py           可被 joblib 反序列化的模型类（稳定 pickle 路径）
scripts/predict.py                 加载模型预测（CLI + Python API + 板块路由）
app.py                             Streamlit 网页演示
data/processed/ipo_offline.db      清洗后 SQLite（回测/查询中间层）
outputs/*/models/*.joblib          序列化模型 + 特征列表
outputs/factor_insights/           因子字典、IC、五分位分组、SHAP贡献、领导速览报告和图表
```

### 后续待办

> 详细的待完善增强项（市场流动性特征、SHAP 解释、增量数据入库、按名称预测）见 `需要完善的部分.md`。

- [~] 主板/北交所精度优化：数据已基本齐全，受样本量（主板约 105、北交所有标签 41）与板块机制限制，进一步提升空间有限，暂缓、不作为优先项。
- [ ] 增加策略层评估：按预测中签率排序的申购优先级、分档命中率、模拟收益。
- [ ] 在网页中加入历史相似新股检索与影响因素解释区。
- [ ] 形成 Word 课题报告与 PPT 答辩稿（汇报文档已在 `项目汇报_新股网下中签率预测.md`）。
- [ ] 录制产品 demo（`streamlit run app.py` 后录屏，覆盖按名称查询/手动输入/三阶段切换，产出 demo.mp4 + 关键截图）。
- [ ] 升级 Anaconda 环境 jinja2（≥3.1.2）或继续使用 `column_config` 规避 Styler 依赖。
