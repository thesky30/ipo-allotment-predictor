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

> **当前状态**：数据清洗、三阶段（T-6 / T-1 / T+1）建模、扩张窗口时间序列回测与 Streamlit 网页演示均已完成并开箱即用。正式生产口径为 **T-6 LightGBM（询价前）**，OOS Spearman ≈ 0.62。详见 [技术架构与模型结果](#技术架构与模型结果)。

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

> 实际落地：正式标签即 `log_offline_oversubscription`，预测后由 `1 / exp(y_pred)` 反推中签率。备选目标均属 T+2 申购/配售结果，仅用于构造标签或事后对照，**不作为模型输入**（见 [因子字典](#因子字典特征全集)）。

## 样本范围

本项目重点关注注册制之后的新股样本，注册制前后切分已完成，正式建模仅用注册制后样本。

| 板块 | 样本口径 | 当前状态 |
| --- | --- | --- |
| 主板 | 2023 年全面注册制之后 | 已切分；有标签样本约 **105 条**，样本量偏小、询价前 OOS 偏弱 |
| 创业板 | 2020 年注册制改革之后 | 已切分；样本量充足，OOS 表现优异 |
| 科创板 | 开板以来（完整注册制） | 样本量最充足，OOS 表现最好 |
| 北交所 | 与其他板块合并训练、单独评估 | 有标签样本约 **41 条**，受机制限制难扩充，未独立建模 |

> 数据落地于 `data/processed/ipo_offline_sample.csv`（约 1.6k 行原始样本，部分无网下标签）；扩张窗口回测的 OOS 评估样本 n=624（切分点 2022/2023/2024/2025）。

## 数据字段分层

> 本节为字段口径的概念性分层；逐因子的完整清单、来源与预期方向见下文 [因子字典（特征全集）](#因子字典特征全集)。

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

> 下列五步技术路线 **均已完成落地**，以下保留原始规划并标注 ✅ 实际产出与对应脚本/目录。

### 1. 取数据 ✅

从 Wind 整理注册制后新股发行样本，已完成：

- ✅ 导出主板、创业板、科创板、北交所 IPO 样本。
- ✅ 按板块和注册制时间切分样本。
- ✅ 建立原始字段字典（`outputs/initial_analysis/field_dictionary.csv`）。
- ✅ 标记字段发布时间和预测时点可用性（三阶段 `FEATURE_NODES`）。
- ✅ 保存原始数据（`data/raw/`，gitignore）、清洗数据（`data/processed/`）、建模数据（`ipo_offline.db`）三个层级。

### 2. 分析数据 ✅

已完成（产出见 `outputs/initial_analysis/` 与 `outputs/factor_insights/`）：

- ✅ 样本数量统计、字段缺失率统计（`missing_by_field.csv`）。
- ✅ 标签分布分析、分板块对比分析（`board_summary.csv`）。
- ✅ 异常值检查（`outlier_checks.csv`）。
- ✅ 特征与标签相关性分析（`correlation_*.csv`）。
- ✅ 注册制前后样本差异分析。

重点结论：

- 网下超额认购倍数 **随板块显著不同**：科创板/创业板 OOS 远高于主板。
- 询价申购总量、询价/配售对象家数与中签率强相关，但这些字段属 **T-1**（询价后），不进入正式模型。
- 战略配售占比已纳入 **T-6** 因子（`strategic_allocation_share_pct`）。
- 北交所样本仅约 41 条，**无需也无法独立建模**，改为合并训练、单独评估。

### 3. 建立多种预测模型 ✅

第一阶段模型（候选，已实现并对比）：

```text
板块均值模型 (board_mean_t6)        ← 已实现，作 T-6 基准
Ridge                              ← 已实现 (ridge_t1)
LightGBM                           ← 已实现，最终生产模型
```

第二阶段模型：

```text
统一模型 + 板块特征   ← 已实现，当前生产方案
板块专属模型          ← 已实现 (board_models.py)，未优于统一模型
全市场模型 + 板块残差校准   ← 未实现，列入 backlog
相似新股检索 + 机器学习预测融合  ← 未实现，列入 backlog
```

> **结论**：LightGBM 显著优于 Ridge（关系高度非线性）；统一模型（含 `board` 特征 + 跨板块学习）≥ 板块专属模型。详见 [回测结果](#回测结果扩张窗口-oos按申购截止日排序)。

### 4. 搭建回测系统 ✅

回测系统已实现（`baseline_models.py`），模拟真实预测过程、避免未来函数：

- ✅ 按 `subscription_deadline_date`（申购截止日）排序。
- ✅ 使用过去样本训练、预测未来样本。
- ✅ 采用扩张窗口（切分点 2022/2023/2024/2025）。
- ✅ 分板块、分年度输出误差指标（`metrics_by_board.csv`、`metrics_by_year.csv`）。
- ✅ 比较 T-6 / T-1 / T+1 多模型预测能力。

已采用的评价指标（产出 `outputs/baseline_models/`）：

```text
MAE
R2
Spearman 排名相关
分板块误差
分年度误差
```

> 真实无泄漏成绩只看 `outputs/baseline_models/predictions.csv`。

### 5. 部署到网页 ✅

Streamlit 网页工具（`app.py`）已实现，包括：

- ✅ 新股信息输入区（按代码/名称查询，或手动输入字段）。
- ✅ 手动输入页可上传「发行安排及初步询价公告」PDF 回填发行结构/申购规则，也可另传招股书 PDF 补全营收、3 年营收 CAGR、可比 PE、拟募资额；PDF 抽取结果必须人工核对后才预测。
- ✅ 预测结果展示区。
- ✅ 特征贡献或影响因素解释区。
- ✅ 模型版本和样本范围说明。
- ⏳ 历史相似新股展示区（列入 backlog）。

核心输出：

```text
预测网下超额认购倍数
预测网下中签率
主要影响因素（特征贡献）
```

> 运行：`streamlit run app.py`（依赖已提交的 `ipo_offline.db` 与 `outputs/.../models/`，开箱即用）。

## 板块建模问题

项目研究的核心问题之一：主板、创业板、科创板、北交所应使用统一模型，还是分板块建模。**该问题已通过时间序列回测得出明确结论（见下文「实际结论」）。**

### 可选方案

| 方案 | 描述 | 优点 | 风险 |
| --- | --- | --- | --- |
| 全市场大模型 | 所有板块合并训练，加入板块特征 | 样本最多，第一版实现简单 | 可能忽略板块差异 |
| 分板块模型 | 每个板块单独训练模型 | 解释清晰，尊重制度差异 | 部分板块样本少 |
| 全市场模型 + 板块校准 | 先训练统一模型，再按板块修正预测残差 | 兼顾样本量和差异 | 实现和验证更复杂 |
| 分层模型 | 共享全市场信息，同时学习板块差异 | 理论上更稳健 | 对实现和样本量要求更高 |

### 实际结论

三条线均已实现并经扩张窗口回测对比：

```text
统一模型
统一模型 + 板块特征   ← 当前生产方案
板块专属模型
```

- **统一模型 + 板块特征 + 跨板块学习 ≥ 板块专属模型**：主板样本少（约 105 条），单独建模反而更差，不要默认"分板块就更好"。
- 北交所有标签样本仅约 41 条、受机制限制难扩充，**未独立建模**，改为与其他板块合并训练、单独评估。
- 最终生产口径即统一 **T-6 LightGBM**（含 `board` 板块特征 + 跨板块学习）。

## 预期成果

项目应形成的成果与当前状态：

- ⏳ 一份 Word 课题报告（草稿见 `项目汇报_新股网下中签率预测.md`）。
- ⏳ 一份 PPT 答辩展示（可由 `scripts/md_to_revealjs.py` 生成）。
- ✅ 一套数据处理流程（`scripts/initial_data_analysis.py` 等）。
- ✅ 一套模型训练流程（`scripts/baseline_models.py`）。
- ✅ 一套时间序列回测系统（扩张窗口，产出 `outputs/baseline_models/`）。
- ✅ 一个可演示的网页预测工具（`app.py`）。
- ✅ 一份 AI 交互记录和开发过程说明（`项目汇报_新股网下中签率预测_transcript.md`）。

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
- [x] 询价前因子洞察（IC / 分组收益 / SHAP / 板块时期画像，`factor_insights.py`）。
- [x] 完整因子字典写入 README（T-6 / T-1 / T+1 / T+2 全集与预期方向）。

> 剩余增量项（增量数据入库、录制 demo、公网部署、板块残差校准、相似新股检索等）见 `需要完善的部分.md`。

## 技术架构与模型结果

### 三阶段时点框架

为彻底杜绝未来数据泄露，所有特征按“信息释放时点”打标签，并据此构造三个模型：

| 阶段 | 时点 | 可用信息 | 用途 |
| --- | --- | --- | --- |
| **T-6** | **询价前正式预测** | 发行结构、申购规则、行业 PE、市场流动性/情绪、批次竞争、同板块破发率、市值门槛、预计募资额、营收及 CAGR、板块滚动行情、主承销商历史表现、行业历史 IPO 热度 | **当前网页默认模型** |
| T-1 | 询价后研究对照 | T-6 全部 + 询价结果（询价超额认购倍数、机构家数、报价分布、最终发行价） | 信息增益分析 / 历史对照 |
| T+1 | 回拨后研究对照 | T-1 全部 + 回拨比例 | 事后校验 / 上界参考 |
| T+2 | — | 网下超额认购倍数（目标变量） | 永不作为输入 |

正式预测节点为"询价开始前"。网页和 CLI 默认使用 **T-6 LightGBM**，不使用询价结果、网下申购、配售、回拨或上市后数据。T-1 / T+1 仅作为研究对照，用于展示信息逐步释放后的预测上界。

### 因子字典（特征全集）

所有特征的时点归属由 `scripts/baseline_models.py` 的 `FEATURE_NODES` 字典唯一定义，因子方向与计算口径见 `outputs/factor_insights/factor_dictionary.csv`。**新增任何字段前必须先判定其所属阶段**，判断错误即构成数据泄露。

#### ✅ T-6 询价前（唯一正式生产口径，约 42 个因子）

正式模型 `lgbm_t6` 仅使用以下询价开始前即可获得的因子，按因子组分类：

| 因子组 | 字段 | 来源 | 预期方向 |
| --- | --- | --- | --- |
| 板块结构 | `board` | 常识/招股书 | 板块制度差异 |
| 发行结构 | `total_issue_shares_10k` | 招股书 | 发行规模越大，供给越多，中签率可能更高 |
| 发行结构 | `offline_issue_before_clawback_10k` | 发行安排公告 | 网下初始发行量越大，中签率可能更高 |
| 发行结构 | `online_issue_before_clawback_10k` | 发行安排公告 | 网上初始发行量，影响回拨前结构 |
| 发行结构 | `offline_issue_before_share_pct` | 派生 | 网下初始占比越高，供给越充分 |
| 战略配售 | `strategic_allocation_10k` | 发行安排公告 | 战略配售绝对量 |
| 战略配售 | `strategic_allocation_share_pct` | 派生 | 战略配售占比高可能压缩可网下分配供给 |
| 申购规则 | `subscription_upper_limit_10k` | 询价及推介公告 | 申购上限影响机构可申购规模和拥挤度 |
| 申购规则 | `subscription_lower_limit_10k` | 询价及推介公告 | 申购下限反映参与门槛 |
| 申购规则 | `subscription_step_10k` | 询价及推介公告 | 申购步长影响报价/申购离散度 |
| 申购规则 | `offline_market_value_threshold_10k_yuan` | 询价公告 | 网下市值门槛，影响可参与机构数 |
| 发行定价（科创板） | `offer_price_upper_yuan` / `offer_price_lower_yuan` | 询价公告 | 价格区间上/下限（多数板块数据缺失） |
| 发行定价（科创板） | `offer_price_range_pct` | 派生 | 询价区间宽度反映定价不确定性 |
| 发行规模 | `expected_fundraising_100m_yuan` / `log_expected_fundraising` | 询价公告/招股书 | 预计募资额（及对数） |
| 公司规模 | `latest_revenue_100m_yuan` / `log_latest_revenue` | 招股书财务摘要 | 近一年营收（及对数） |
| 成长 | `revenue_cagr_3y_pct` | 招股书财务摘要 | 三年营收复合增速 |
| 估值 | `industry_pe_at_ipo` | 市场数据 | 行业估值环境影响新股吸引力 |
| 估值 | `comparable_pe_avg_ex_nonrecurring` | 招股书 | 可比公司扣非估值影响询价前吸引力判断 |
| 市场流动性 | `market_turnover_ma20` | 市场成交额20日均值 | 流动性越高，资金越活跃，可能推高超额认购、压低中签率 |
| 市场流动性 | `market_turnover_pct_rank_1y` | 成交额20日均值1年滚动分位 | 高分位流动性对应更拥挤的网下申购 |
| 市场流动性 | `market_turnover_ma20_over_ma60` | 成交额短/长均线比 | 短期成交额放大代表情绪升温 |
| 市场情绪 | `market_return_ma20` | 市场指数20日涨跌幅 | 市场上涨可能增强申购热度 |
| 市场热度 | `recent_ipo_first_day_return_ma20` | 同板块近20只已上市IPO首日涨幅 | 赚钱效应越强，申购越拥挤 |
| 市场热度 | `same_board_break_rate_ma10` | 同板块近10只已上市IPO破发比例 | 破发率越高，申购意愿可能下降 |
| 批次竞争 | `concurrent_ipo_count` | 申购日历±7天全市场IPO数 | IPO越拥挤可能分流资金 |
| 批次竞争 | `same_board_concurrent_ipo_count` | 申购日历±7天同板块IPO数 | 同板块供给拥挤分流同类资金 |
| 批次竞争 | `concurrent_offline_issue_sum_10k` | 申购日历±7天其他IPO网下发行量合计 | 同窗口可申购供给越大，单只拥挤度缓和 |
| 板块流动性 | `board_turnover_ma20` / `board_turnover_pct_rank_1y` / `board_turnover_ma20_over_ma60` | 板块行情严格向前滚动 | 板块成交额均值/分位/均线比 |
| 板块情绪 | `board_return_ma20` | 板块行情严格向前滚动 | 板块20日涨跌幅 |
| 承销商声誉 | `underwriter_prior_ipo_count` | 主承销商历史已发生IPO | 主承销商历史发行经验 |
| 承销商声誉 | `underwriter_prior_log_oversub_mean` | 主承销商历史已发生IPO | 主承销商历史超额认购均值 |
| 承销商声誉 | `underwriter_prior_first_day_return_mean` | 主承销商历史已上市IPO | 主承销商历史首日涨幅 |
| 承销商声誉 | `underwriter_prior_break_rate` | 主承销商历史已上市IPO | 主承销商历史破发率 |
| 行业历史热度 | `sw_l1_prior_ipo_count` | 申万一级历史已发生IPO | 行业历史发行数量 |
| 行业历史热度 | `sw_l1_prior_log_oversub_mean` | 申万一级历史已发生IPO | 行业历史超额认购均值 |
| 行业历史热度 | `sw_l1_prior_first_day_return_mean` | 申万一级历史已上市IPO | 行业历史首日涨幅 |
| 行业历史热度 | `sw_l1_prior_break_rate` | 申万一级历史已上市IPO | 行业历史破发率 |

> T-6 因子覆盖六大维度：**发行结构与申购规则、估值与公司基本面、市场流动性/情绪/热度、批次竞争、板块滚动行情、承销商与行业历史声誉**。其中 `market_turnover_ma20`（市场流动性）特征重要性排名第 2。

#### ⚠️ T-1 询价后（仅研究对照，禁止进入正式模型）

下列字段在询价结果公布后才可得，含未来信息。`lgbm_t1` 的 OOS Spearman ≈ 0.96 **是因为偷看了询价/定价结果**，不能当作正式预测效果。

| 因子组 | 字段 | 说明 |
| --- | --- | --- |
| 发行定价 | `offer_price_yuan` | 最终发行价 |
| 发行定价 | `offer_price_position_in_range`（科创板） | 发行价在询价区间中的位置 |
| 估值（依赖最终价） | `ipo_pe_diluted` / `issue_pb` / `pe_vs_industry` / `pe_vs_comparable` | 基于最终发行价的估值 |
| 发行规模（依赖最终价） | `issue_amount_100m_yuan` | 募集资金总额 |
| 询价结果 | `inquiry_subscription_total_10k` / `inquiry_investors_count` / `inquiry_allotment_accounts` / `inquiry_oversubscription_ratio` | 初步询价申购总量、询价/配售对象家数、询价超额认购倍数 |
| 询价价格 | `quote_price_weighted_avg` / `quote_price_median` / `quote_price_vs_offer` | 报价加权均值/中位数及与发行价之比 |
| 询价结果 | `excluded_subscription_share_pct` / `high_price_excluded_subscription_share_pct` | 剔除无效/最高报价后申购占比 |

#### ⚠️ T+1 回拨后（仅研究对照）

| 因子组 | 字段 | 说明 |
| --- | --- | --- |
| 回拨 | `clawback_ratio_pct` | 回拨比例 |
| 回拨 | `offline_issue_final_10k` / `offline_issue_final_share_pct` | 回拨后网下最终发行量及占比 |
| 回拨 | `online_issue_final_10k` | 回拨后网上最终发行量 |

#### 🚫 T+2 申购/配售结果（目标变量本身，永不作为输入）

`offline_oversubscription_ratio`（及其对数 `log_offline_oversubscription`，即预测标签）、`offline_allotment_ratio_pct`、`offline_subscription_total_10k`、`offline_valid_quote_subscription_10k`、`offline_allotment_accounts`、`offline_inquiry_investors`、`a_investor_*`、`offline_oversubscription_ratio_before_clawback`。

#### 已知暂未纳入的因子（及原因）

| 因子 | 不纳入原因 |
| --- | --- |
| `issue_pb_factor` | 可能依赖最终发行价，有数据泄露风险 |
| 发行价格区间（`offer_price_upper/lower`，非科创板） | 数据全空 |
| 行业行情滚动因子 | 缺申万代码-名称映射，暂无法严格向前滚动 |

### 回测结果（扩张窗口 OOS，按申购截止日排序）

| 模型 | 阶段 | n | MAE | R² | Spearman |
| --- | --- | --- | --- | --- | --- |
| board_mean_t6 | T-6 | 624 | 0.370 | 0.529 | 0.270 |
| lgbm_t6 | T-6 | 624 | 0.307 | 0.666 | 0.619 |
| ridge_t1 | T-1 | 624 | 0.388 | -0.241 | 0.671 |
| lgbm_t1（研究对照） | T-1 | 624 | 0.109 | 0.913 | 0.961 |
| lgbm_t1plus | T+1 | 624 | 0.102 | 0.920 | 0.965 |

- 正式模型（T-6）OOS Spearman 为 0.619；T-1/T+1 仅作为研究对照，不作为正式预测口径。
- 分板块 OOS Spearman（T-1）：科创板 0.986、创业板 0.979、北交所 0.886（n=6）、主板 0.640。
- 板块专属模型未优于统一模型，统一模型 + 板块特征 + 跨板块学习为当前最优方案。
- T-6 正式模型使用的全部因子（约 42 个，含来源与预期方向）及暂未纳入因子，见上文 [因子字典（特征全集）](#因子字典特征全集)。其中 `market_turnover_ma20`（市场流动性）特征重要性排名第 2。

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
scripts/convert_wind_market.py           Wind 市场日度导出 → market_daily.csv（市场流动性特征）
scripts/initial_data_analysis.py         数据清洗 + EDA
scripts/baseline_models.py               三阶段建模 + 时间序列回测
scripts/board_models.py                  板块专项模型对比
scripts/factor_insights.py               询价前因子洞察 + 板块/时期特征报告
scripts/pdf_extract.py                   发行安排及初步询价公告 PDF → T-6 原始字段
scripts/prospectus_extract.py            招股书 PDF 相关页定位 → T-6 财务/估值字段
scripts/process_company_factors.py       公司基本面因子处理
scripts/process_board_industry_market.py  板块/行业/承销商滚动因子
scripts/build_new_factor_research.py     新因子 IC、分组收益分析
scripts/predict.py                       模型加载与预测（CLI / API / 板块路由）
scripts/model_classes.py                 可被 joblib 反序列化的模型类
app.py                                   Streamlit 网页演示
data/processed/ipo_offline.db            清洗后 SQLite（回测/查询层）
outputs/initial_analysis/                EDA 报告、字段字典、相关性、缺失率
outputs/baseline_models/                 回测指标、特征重要性、序列化模型、报告
outputs/board_models/                    板块对比指标与报告
outputs/factor_insights/                 因子IC、分组、SHAP贡献、板块时期画像和领导速览报告
outputs/new_factor_research/             新增因子研究（IC、分组收益）
需要完善的部分.md                        后续开发 backlog（待完善增强项）
```

> 后续待办项（增量数据入库、录制 demo、公网部署等）详见 `需要完善的部分.md`。

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
- `outputs/factor_insights/`：因子 IC、SHAP 贡献、板块时期画像和领导速览报告。
- `outputs/new_factor_research/`：新增因子研究产出。
- `README.md`、`AGENTS.md`、`需要完善的部分.md`：项目说明、协作规则和 backlog。
- `requirements.txt`：Python 依赖。
- `.python-version`：固定 Python 3.11（与 devcontainer 一致）。

> **部署 Python 版本（重要）**：模型 `.joblib` 在 Python 3.11/3.13 + sklearn 1.6.1 下保存，**部署必须用 Python 3.11**，否则会出问题。Streamlit Community Cloud 默认会拉最新 Python（如 3.14），而该 ML 栈在 3.14 上没有预编译 wheel —— uv 会改为源码编译且环境可能残缺，表现为运行时 `ModuleNotFoundError: joblib`（实为整套依赖没装全）。
> 修复：① 仓库已含 `.python-version=3.11`；② **Streamlit Cloud 后台 Manage app → Settings → Python version 选 3.11**（这是权威设置），保存后 Reboot。

不建议提交：

- Wind/Tushare 原始导出 Excel、个人下载路径和未脱敏数据。
- `.venv/`、`__pycache__/`、`.claude/`、编辑器配置和本地日志。
- `outputs/peek_*.json` 等临时探查文件。
- `reveal/`、`*_slides.html`、一次性 HTML 构建产物。

### 新机器启动

```bash
git clone https://github.com/thesky30/ipo-allotment-predictor.git
cd ipo-allotment-predictor

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
git remote add origin https://github.com/thesky30/ipo-allotment-predictor.git
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
