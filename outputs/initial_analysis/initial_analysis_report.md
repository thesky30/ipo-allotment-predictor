# 初步数据处理与分析报告

生成日期：2026-05-21

## 1. SQL / SQLite 是否值得用

值得用，但定位应是“中间层和回测查询层”，不是替代 pandas / sklearn 的建模层。

- 当前 3 个 Excel 合计 1,326 行、字段不多，单机 pandas 已经足够快。
- SQLite `.db` 的价值在于：统一字段名、保留原始来源、方便按板块/年份/时间窗口滚动取数、让后续网页或回测服务直接查询。
- 建议流程：Excel 原始层 -> 清洗后的 `ipo_offline_sample` 表 -> 特征/标签视图 -> pandas/sklearn 建模。
- 不建议一开始把所有特征工程都写进 SQL；复杂统计、缺失处理、分位裁剪、时间序列回测仍用 Python 更顺手。

本次已生成 SQLite 数据库：`data/processed/ipo_offline.db`，核心表为 `ipo_offline_sample`。

如环境安装了 matplotlib，脚本会在 `outputs/initial_analysis/figures/` 额外生成可视化图片；当前 bundled Python 未包含该依赖时会自动跳过。

## 2. 样本概况

- 总样本：1,326 条。
- 有网下超额认购倍数标签：1,194 条，占 90.0%。
- 上市日期范围：2019-07-22 至 2026-05-18。
- 当前覆盖：科创板、注册制创业板、主板注册制后样本；尚未包含北交所。

| board | sample_count | label_count | date_min | date_max | median_offline_oversubscription | mean_offline_oversubscription | median_log_oversubscription | median_offline_allotment_ratio_pct | median_offline_subscription_total_10k | median_offline_accounts | median_inquiry_investors | median_issue_amount_100m_yuan | missing_label_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 主板 | 112 | 105 | 2023-04-10 | 2026-05-18 | 8,622.80 | 8,376.71 | 9.06 | 0.01 | 8,314,550.00 | 7,437.00 | 539.00 | 9.17 | 7 |
| 创业板 | 603 | 481 | 2020-08-24 | 2026-05-11 | 3,367.41 | 3,426.11 | 8.12 | 0.03 | 5,496,420.00 | 6,351.00 | 276.00 | 7.05 | 122 |
| 科创板 | 611 | 608 | 2019-07-22 | 2026-04-24 | 2,596.93 | 2,713.03 | 7.86 | 0.04 | 4,700,370.00 | 6,287.00 | 288.50 | 10.08 | 3 |

## 3. 年份与板块分布

| listing_year | 主板 | 创业板 | 科创板 |
| --- | --- | --- | --- |
| 2,019 | 0 | 0 | 70 |
| 2,020 | 0 | 63 | 143 |
| 2,021 | 0 | 199 | 162 |
| 2,022 | 0 | 150 | 124 |
| 2,023 | 36 | 110 | 67 |
| 2,024 | 24 | 38 | 15 |
| 2,025 | 38 | 33 | 19 |
| 2,026 | 12 | 8 | 9 |

## 4. 标签描述性统计

核心标签为 `log_offline_oversubscription = log(网下超额认购倍数)`。

| field | count | missing | missing_rate | mean | std | min | p01 | p25 | median | p75 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline_oversubscription_ratio | 1,194 | 132 | 0.0995 | 3,498.3550 | 2,113.7684 | 107.9693 | 293.3546 | 2,231.4206 | 3,165.6802 | 4,266.9856 | 10,746.8098 | 13,987.6532 |
| log_offline_oversubscription | 1,194 | 132 | 0.0995 | 7.9643 | 0.6944 | 4.6818 | 5.6813 | 7.7104 | 8.0601 | 8.3587 | 9.2824 | 9.5459 |
| offline_allotment_ratio_pct | 1,194 | 132 | 0.0995 | 0.0485 | 0.0671 | 0.0071 | 0.0093 | 0.0234 | 0.0316 | 0.0448 | 0.3409 | 0.9262 |
| offline_oversubscription_ratio_before_clawback | 1,194 | 132 | 0.0995 | 2,463.2113 | 1,061.0232 | 94.4700 | 256.8622 | 1,770.3234 | 2,436.5329 | 3,171.4300 | 4,805.9794 | 6,339.6800 |
| a_investor_lottery_rate_pct | 1,194 | 132 | 0.0995 | 0.0547 | 0.0712 | 0.0072 | 0.0103 | 0.0288 | 0.0376 | 0.0522 | 0.3583 | 1.0471 |

## 5. 可能对预测有用的规律

### 预测前较可能可用字段的相关性

以下仅是单变量相关，不代表因果，也没有经过严格时间序列验证：

| feature | n | spearman_corr | pearson_corr |
| --- | --- | --- | --- |
| a_investor_allotted_accounts | 1,194 | 0.7470 | 0.6998 |
| offline_issue_final_share_pct | 1,194 | -0.4896 | -0.5290 |
| ipo_pe_diluted | 1,130 | -0.4849 | -0.1827 |
| excluded_subscription_share_pct | 1,194 | -0.4725 | -0.3743 |
| issue_amount_100m_yuan | 1,194 | -0.4632 | -0.3379 |
| issue_pb | 1,191 | -0.4628 | -0.3490 |
| a_investor_subscription_share_pct | 1,194 | -0.4336 | -0.4095 |
| pe_vs_industry | 1,127 | -0.4149 | -0.1614 |
| a_investor_subscription_shares_10k | 1,194 | 0.3976 | 0.0852 |
| offer_price_yuan | 1,194 | -0.3487 | -0.2273 |
| a_investor_allotted_shares_10k | 1,194 | -0.3045 | -0.2167 |
| pe_vs_comparable | 1,122 | -0.2476 | -0.1560 |

### 全字段相关性，包括明显事后字段

这些字段可用于理解标签形成机制，但正式预测时要谨慎排除泄露：

| feature | n | spearman_corr | pearson_corr |
| --- | --- | --- | --- |
| offline_allotment_ratio_pct | 1,194 | -1.0000 | -0.8151 |
| implied_offline_lottery_rate_pct | 1,194 | -1.0000 | -0.8151 |
| offline_oversubscription_ratio | 1,194 | 1.0000 | 0.8649 |
| a_investor_lottery_rate_pct | 1,194 | -0.9612 | -0.8035 |
| offline_oversubscription_ratio_before_clawback | 1,194 | 0.9338 | 0.8615 |
| offline_allotment_accounts | 1,194 | 0.7752 | 0.7576 |
| a_investor_allotted_accounts | 1,194 | 0.7470 | 0.6998 |
| offline_inquiry_investors | 1,194 | 0.6936 | 0.6512 |
| offline_issue_final_share_pct | 1,194 | -0.4896 | -0.5290 |
| ipo_pe_diluted | 1,130 | -0.4849 | -0.1827 |
| excluded_subscription_share_pct | 1,194 | -0.4725 | -0.3743 |
| offline_valid_quote_subscription_10k | 1,194 | 0.4651 | 0.1231 |

## 6. 有趣的数据结论

- 板块差异很强：主板 网下超额认购倍数中位数最高 (8,622.80 倍)，科创板 最低 (2,596.93 倍)。
- 在较像预测前可用的字段里，`a_investor_allotted_accounts` 与标签的 Spearman 相关绝对值最高，为 0.747，方向为正相关。
- `网下申购配售比例` 与 `100 / 网下超额认购倍数` 基本互为倒数，二者差异的 95% 分位仅 0.000000 个百分点；建模时二者只能作为标签/校验，不能同时做输入。
- 配售对象家数越多，竞争越拥挤：`offline_allotment_accounts` 与 log 超额认购倍数的 Spearman 约 0.775。但这个字段更像申购后结果，需确认预测时点。

## 7. 缺少哪些数据

当前 Excel 缺少项目文档中许多“初步询价结束后、网下申购前”更理想的输入字段，尤其是：

- 网下申购数量上限、下限、步长。
- 初步询价申报价格明细、加权平均数、中位数。
- 初步询价申报数量、初步询价配售对象家数、询价对象家数。
- 初步询价申购总量、初步询价申购倍数（回拨前）。
- 网下询价市值门槛及 A 类/主题与战略门槛。
- 发行价格下限/上限。
- 剔除无效和最高报价后的申购总量、配售对象、询价对象。
- 主承销商战略获配股份数/占比。
- 行业、保荐机构/主承销商、发行日/申购日、发行阶段市场热度等上下文变量。

当前字段缺失率最高的字段如下：

| field | missing_count | missing_rate |
| --- | --- | --- |
| other_issue_10k | 1,323 | 0.9977 |
| strategic_allocation_10k | 451 | 0.3401 |
| strategic_allocation_share_pct | 451 | 0.3401 |
| offline_issue_before_clawback_10k | 132 | 0.0995 |
| offline_issue_final_10k | 132 | 0.0995 |
| high_price_excluded_subscription_share_pct | 132 | 0.0995 |
| excluded_subscription_share_pct | 132 | 0.0995 |
| offline_allotment_ratio_pct | 132 | 0.0995 |

## 8. 异常值与数据质量提示

| field | p01 | p99 | below_p01_count | above_p99_count | min_security | min | max_security | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline_oversubscription_ratio | 293.3546 | 10,746.8098 | 12 | 12 | 金山办公 | 107.9693 | 埃泰克 | 13,987.6532 |
| offline_allotment_ratio_pct | 0.0093 | 0.3409 | 12 | 12 | 埃泰克 | 0.0071 | 金山办公 | 0.9262 |
| offline_subscription_total_10k | 532,944.3000 | 57,513,059.6000 | 12 | 12 | 心脉医疗 | 262,930.0000 | 和辉光电-U | 198,700,240.0000 |
| offer_price_yuan | 3.5671 | 164.3538 | 14 | 14 | 龙腾光电 | 1.2200 | 禾迈股份 | 557.8000 |
| issue_amount_100m_yuan | 2.0465 | 74.8760 | 14 | 14 | 读客文化 | 0.6202 | 中芯国际 | 532.3019 |
| ipo_pe_diluted | 10.7905 | 376.2190 | 13 | 13 | 海博思创 | 6.1400 | 孚能科技 | 1,737.4900 |
| clawback_ratio_pct | 0.0000 | 40.0000 | 0 | 0 | 利安科技 | 0.0000 | 世盟股份 | 40.0000 |

## 9. 建模建议

- 第一版基线可以用 `log_offline_oversubscription` 做标签，并同时保留 `board` 类别特征。
- 先建立三个不泄露的基线：板块滚动均值、年份/板块滚动均值、只用发行规模/价格/PE/战略配售/剔除比例的 Ridge 或树模型。
- `offline_subscription_total_10k`、`offline_allotment_accounts`、`offline_allotment_ratio_pct` 等字段预测力很强，但多数属于申购后/配售后信息，应作为标签或事后解释，不应进入正式预测输入。
- 科创板、创业板、主板中位水平差异较明显，后续必须分板块评估；统一模型要加入板块特征，且建议做板块残差校准。
- 下一步最关键不是换模型，而是补齐真正预测时点可见的初步询价字段。
