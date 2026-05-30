# 未入库新股的询价前预测（Phase 1 + Phase 2）设计文档

- 日期：2026-05-31
- 状态：已通过 brainstorming，待用户复核 → writing-plans
- 范围：Phase 1（冷启动预测核心）+ Phase 2（巨潮 PDF → LLM 抽取 → 回填表单）
- 不在本轮范围：Phase 3 爬虫、预测区间/不确定性建模、数据库持续自动更新

## 1. 背景与目标

现有网页主要服务「已入库历史股票」的查询，其逐股结果是样本内、偏乐观。真实使用场景是：**一只尚未入库、尚未询价的新股**，在「初步询价开始前」只有招股书/发行公告里的少量基础字段（募资额、公司 PE、营收、行业 PE、发行结构、申购规则等）。

目标：让网页支持对**任意未入库新股**，用询价前可得的原始字段做 T-6 正式预测，并**诚实呈现**——因为这类股票尚无真实披露的网下中签率，无法计算「本股准确率」。

预测对象不变：`y = log(网下超额认购倍数)`，中签率由 `1 / exp(y_pred)` 反推；正式口径为 T-6 LightGBM。

## 2. 关键决策（brainstorming 结论）

1. **上下文因子全自动补算**：用户只填招股书/公告原始字段；21 个上下文因子（见 §4③）由系统按「申购截止日」从参考表自动算出。
2. **无标签诚实展示**：不展示「本股准确率」；改为展示预测值 + 模型整体回测水平（标注为模型级）+ 同板块历史分位 + 因子贡献，并显式说明本股暂无真实披露。
3. **本轮范围 = Phase 1 + Phase 2**；爬虫（Phase 3）后续单独出 spec。
4. **数据新鲜度 = 加市场数据刷新步骤**，但刷新源是**可编程开放源（Tushare 主 / AkShare 备），不是 Wind**（Wind 无持续 API 权限，详见 §9）。
5. **PDF 抽取 = LLM 方案，且 provider-agnostic**：通过 OpenAI 兼容客户端（base_url + key + model 配置化）接入用户自备 API；抽取结果须经人工确认表单后才预测。

## 3. 整体架构与模块边界

新增 5 个纯逻辑模块（不依赖 Streamlit、不在导入期联网），`app.py` 仅做装配：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `scripts/reference_data.py` | 加载市场/板块/历史 IPO 参考表，提供「≤ 某日期最近一条」访问器，返回 `data_as_of` | 只读 csv / `ipo_offline.db` |
| `scripts/feature_assembly.py` | 原始字段 dict + 申购截止日 → 完整 42 维 T-6 向量 + 元数据（data_as_of、warnings） | reference_data |
| `scripts/market_source.py` | 可编程开放源（Tushare 主 / AkShare 备）增量刷新参考表 | 复用 `fetch_market_data.py` |
| `scripts/llm_client.py` | OpenAI 兼容薄客户端（base_url/key/model 来自 env / Streamlit secrets） | 用户自备 API |
| `scripts/pdf_extract.py` | 巨潮 PDF → 取文本 → LLM 按 JSON schema 抽取字段 dict | pdfplumber、llm_client |

数据流：

```
PDF →(pdf_extract)→ 字段 dict →[人工核对/可编辑表单]→(feature_assembly)→ 42维向量 →(predict.predict_from_dict)→ 结果展示
                                         ↑ 手动输入也走同一表单（Phase 1 地基）
```

## 4. 输入 schema（42 个 T-6 因子的来源分类）

### ① 用户 / PDF 提供（16 个模型特征 + 3 个键）

模型特征：`board`、`total_issue_shares_10k`、`offline_issue_before_clawback_10k`、`online_issue_before_clawback_10k`、`strategic_allocation_10k`、`subscription_upper_limit_10k`、`subscription_lower_limit_10k`、`subscription_step_10k`、`offline_market_value_threshold_10k_yuan`、`industry_pe_at_ipo`、`comparable_pe_avg_ex_nonrecurring`、`expected_fundraising_100m_yuan`、`latest_revenue_100m_yuan`、`revenue_cagr_3y_pct`、`offer_price_upper_yuan`、`offer_price_lower_yuan`（后两者科创板专属，常缺失）。

键（非模型特征，但补算必需）：`subscription_deadline_date`（as-of 锚点）、`underwriter`（主承销商名）、`sw_level1_industry_code`（申万一级行业代码）。

### ② 由 ① 即时派生（5 个）

`offline_issue_before_share_pct`、`strategic_allocation_share_pct`、`offer_price_range_pct`、`log_expected_fundraising`、`log_latest_revenue`。

### ③ 按申购截止日自动补算（21 个上下文因子）

- 市场流动性/情绪（4）← `market_daily`：`market_turnover_ma20`、`market_turnover_pct_rank_1y`、`market_turnover_ma20_over_ma60`、`market_return_ma20`
- 板块滚动（4）← `board_market_daily`：`board_turnover_ma20`、`board_turnover_pct_rank_1y`、`board_turnover_ma20_over_ma60`、`board_return_ma20`
- 同板块热度/破发（2）← 历史 IPO 样本：`recent_ipo_first_day_return_ma20`、`same_board_break_rate_ma10`
- 批次竞争（3）← IPO 日历（±7 天）：`concurrent_ipo_count`、`same_board_concurrent_ipo_count`、`concurrent_offline_issue_sum_10k`
- 承销商先验（4）← 历史样本按 `underwriter`：`underwriter_prior_ipo_count`、`underwriter_prior_log_oversub_mean`、`underwriter_prior_first_day_return_mean`、`underwriter_prior_break_rate`
- 申万行业先验（4）← 历史样本按 `sw_level1_industry_code`：`sw_l1_prior_ipo_count`、`sw_l1_prior_log_oversub_mean`、`sw_l1_prior_first_day_return_mean`、`sw_l1_prior_break_rate`

合计 16 + 5 + 21 = 42，与 `baseline_models.py` 的 `FEATURE_NODES`（T-6）一一对应。

## 5. 核心技术风险：训练/推理特征一致性（train-serve skew）

自动补算的滚动均值、历史先验、批次竞争若与训练时算法不一致，线上向量与模型口径错位，预测会悄悄失真。

**对策：**
1. 将「按时点计算」逻辑从 `initial_data_analysis.py` / `process_company_factors.py` / `process_board_industry_market.py` 抽成**共享函数**，训练建表与 `feature_assembly` **调用同一份实现**；只做提取重构，不改动口径。
2. 新增**一致性测试**：对已入库的若干只股票，用 `feature_assembly` 重算这 21 个上下文因子，与训练表（`ipo_offline.db` / `ipo_offline_sample.csv`）中存储值逐一比对，须在数值容差内相等。

## 6. PDF 抽取（Phase 2）与人工确认

- 锁定**「发行安排及初步询价公告」**（短、字段集中），不解析整本招股书。
- `pdf_extract`：pdfplumber 取文本 → 截取相关段 → 调 `llm_client` 按严格 JSON schema（即 §4 的 ① 字段）返回结构化 dict。
- **强制人工确认**：抽取结果回填到**可编辑表单**，用户核对/改正后才允许预测——挡 LLM 幻觉，也兜底抽取失败。
- `llm_client` 为 OpenAI 兼容协议，配置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，支持用户自备 API（DeepSeek / Qwen / Kimi / Zhipu / OpenAI 等）。
- 未配置 API 或抽取失败：自动降级到纯手填，给清晰提示，绝不在用户未确认字段时自动预测。

## 7. 结果展示（诚实无标签）

结果区：预测超额认购倍数 → 中签率；同板块历史分位；主要因子贡献（复用 `predict.py` 的 explanation / `app.py` 的 `show_explanation`）。显式标注：
- 「本股暂无真实披露，无法计算本股准确率」
- 「以下为**模型整体回测水平**：OOS Spearman 0.62 / MAE 0.31（模型级，非本股）」
- 「市场/参考数据截至 YYYY-MM-DD」（来自 `feature_assembly` 的 `data_as_of`）

## 8. 数据源与刷新

- 「刷新市场数据」按钮调用 `market_source.refresh(as_of)`，源为 **Tushare（主）/ AkShare（备）**，复用 `fetch_market_data.py`。**Wind 不进入持续路径**。
- 未配置 Tushare token 时按钮禁用并提示；刷新失败回退到现有静态快照 + data_as_of 警告。
- 刷新范围（Phase 1）：市场日行情、板块日行情；近月新增 IPO 记录（用于先验/批次竞争/同板块热度）作为可选增强项，若实现不及可留待 Phase 3。

## 9. 答辩说明点（数据获取限制与未来优化）

- **为何未做到持续更新**：训练数据用 Wind 一次性历史导出（字段最全、口径权威）；但 **Wind 无持续 API 权限**是硬约束，无法自动增量更新数据库。
- **现状对策**：数据源做抽象层，生产/持续更新改用可编程开放源（Tushare/AkShare）；Wind 仅用于建模期历史底座。
- **未来优化**：若获得 Wind 量化接口（WindPy）或机构数据服务，数据源可无缝替换；持续 DB 更新、预测区间、爬虫自动取数为后续阶段。
- 「完全没披露的最新股只能给预测、给不了本股准确率」与「上下文因子可能略旧」均由数据获取权限所限，非方法缺陷。

## 10. 错误处理

- 未知承销商 / 新行业（无历史样本）→ 对应先验置缺失（LightGBM 原生处理）+ 页面提示「该承销商/行业无历史样本，先验按缺失」。
- 申购截止日超出参考快照范围 → 用最新可得 ≤ 日期数据 + 明显警告。
- 批次竞争窗口数据不全（未来同期 IPO 未知）→ 标注「可能偏低」。
- PDF 抽取失败 / LLM 未配置 → 降级纯手填。

## 11. 测试策略

- 单元：`feature_assembly` 一致性测试（对训练行重算 21 因子比对）；§4② 派生字段数学单测。
- 单元：`pdf_extract` 金样本测试（LLM 调用 mock，保证 CI 确定性）。
- 集成：对一只已入库股票，用 `feature_assembly` 组装向量后 `predict_from_dict`，结果应与 `predict_from_code` 吻合（验证组装正确）。
- UI：手动冒烟（上传 PDF → 回填 → 预测 → 展示）。

## 12. 非目标 / 未来阶段

- Phase 3：爬虫自动从巨潮抓取发行公告 PDF。
- 预测区间 / 不确定性（分位数模型或残差 bootstrap）。
- 数据库持续自动更新（依赖开放源调度或 Wind 权限获取）。
