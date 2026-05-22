# 初步数据处理与分析报告（v2）

生成日期：2026-05-22

## 变更说明（相对 v1）

- 新增北交所（北交所_网下打新数据.xlsx），样本扩展至四个板块。
- 合并三份补充数据（科创板/创业板/北交所），新增初步询价阶段字段：
  询价申购总量、询价/配售对象家数、申购步长/上下限、申报价格均值/中位数、
  询价截止日、申购截止日、上市首日涨跌幅。
- 科创板补充数据额外包含发行价格上限/下限（底价）。
- 主板暂无补充数据；主板询价字段全为 NaN。
- 北交所大部分样本不适用询价机制；询价字段有效样本仅约 41 条。

## 1. 样本概况

- 总样本：1,643 条。
- 有网下超额认购倍数标签：1,235 条，占 75.2%。
- 有初步询价补充字段：1,235 条。
- 上市日期范围：2019-07-22 至 2026-05-22。

| board | sample_count | label_count | inquiry_field_count | date_min | date_max | median_offline_oversubscription | mean_offline_oversubscription | median_log_oversubscription | median_offline_allotment_ratio_pct | median_inquiry_subscription_total_10k | median_inquiry_investors_count | median_issue_amount_100m_yuan | missing_label_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 主板 | 112 | 105 | 105 | 2023-04-10 | 2026-05-18 | 8,622.80 | 8,376.71 | 9.06 | 0.01 | 8,789,660.00 | 688.00 | 9.17 | 7 |
| 创业板 | 603 | 481 | 481 | 2020-08-24 | 2026-05-11 | 3,367.41 | 3,426.11 | 8.12 | 0.03 | 7,106,380.00 | 370.00 | 7.05 | 122 |
| 北交所 | 317 | 41 | 41 | 2020-07-27 | 2026-05-22 | 21.37 | 29.35 | 3.06 | 4.68 | 29,834.83 | 570.00 | 1.77 | 276 |
| 科创板 | 611 | 608 | 608 | 2019-07-22 | 2026-04-24 | 2,596.93 | 2,713.03 | 7.86 | 0.04 | 6,014,455.00 | 373.00 | 10.08 | 3 |

## 2. 年份与板块分布

| listing_year | 主板 | 创业板 | 北交所 | 科创板 |
| --- | --- | --- | --- | --- |
| 2,019 | 0 | 0 | 0 | 70 |
| 2,020 | 0 | 63 | 38 | 143 |
| 2,021 | 0 | 199 | 40 | 162 |
| 2,022 | 0 | 150 | 83 | 124 |
| 2,023 | 36 | 110 | 77 | 67 |
| 2,024 | 24 | 38 | 23 | 15 |
| 2,025 | 38 | 33 | 26 | 19 |
| 2,026 | 12 | 8 | 28 | 9 |

## 3. 标签描述性统计

核心标签为 `log_offline_oversubscription = log(网下超额认购倍数)`。

| field | count | missing | missing_rate | mean | std | min | p01 | p25 | median | p75 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline_oversubscription_ratio | 1,235 | 408 | 0.2483 | 3,383.1896 | 2,169.3718 | 1.2300 | 13.6334 | 2,119.0459 | 3,087.3255 | 4,222.8902 | 10,730.7212 | 13,987.6532 |
| log_offline_oversubscription | 1,235 | 408 | 0.2483 | 7.7988 | 1.1361 | 0.2070 | 2.6112 | 7.6587 | 8.0351 | 8.3483 | 9.2809 | 9.5459 |
| offline_allotment_ratio_pct | 1,235 | 408 | 0.2483 | 0.3402 | 2.9526 | 0.0071 | 0.0093 | 0.0237 | 0.0324 | 0.0472 | 7.3519 | 81.0437 |
| offline_oversubscription_ratio_before_clawback | 1,235 | 408 | 0.2483 | 2,382.2991 | 1,131.0215 | 1.1700 | 12.2975 | 1,690.6442 | 2,384.8000 | 3,134.9672 | 4,804.1442 | 6,339.6800 |
| a_investor_lottery_rate_pct | 1,194 | 449 | 0.2733 | 0.0547 | 0.0712 | 0.0072 | 0.0103 | 0.0288 | 0.0376 | 0.0522 | 0.3583 | 1.0471 |

## 4. 预测前可用字段相关性（含新增询价字段）

以下排除申购后泄露字段，仅展示可能在预测时点可见的字段：

| feature | n | spearman_corr | pearson_corr |
| --- | --- | --- | --- |
| inquiry_oversubscription_ratio | 1,235 | 0.8004 | 0.7650 |
| a_investor_allotted_accounts | 1,218 | 0.7617 | 0.7527 |
| inquiry_allotment_accounts | 1,235 | 0.5880 | 0.7466 |
| inquiry_investors_count | 1,235 | 0.5053 | 0.1471 |
| excluded_subscription_share_pct | 1,235 | -0.4869 | -0.3393 |
| a_investor_subscription_shares_10k | 1,218 | 0.4325 | 0.1159 |
| ipo_pe_diluted | 1,170 | -0.4205 | -0.0628 |
| offline_issue_final_share_pct | 1,235 | -0.4190 | -0.2772 |
| pe_vs_industry | 1,127 | -0.4149 | -0.1614 |
| issue_pb | 1,232 | -0.4129 | -0.1707 |
| inquiry_subscription_total_10k | 1,235 | 0.4038 | 0.1541 |
| recent_ipo_first_day_return_ma20 | 1,215 | 0.3742 | 0.3102 |
| a_investor_subscription_share_pct | 1,218 | -0.3532 | 0.2232 |
| issue_amount_100m_yuan | 1,235 | -0.3440 | -0.1343 |

## 5. 全字段相关性（含事后字段，供理解机制）

| feature | n | spearman_corr | pearson_corr |
| --- | --- | --- | --- |
| implied_offline_lottery_rate_pct | 1,235 | -1.0000 | -0.5228 |
| offline_oversubscription_ratio | 1,235 | 1.0000 | 0.7237 |
| offline_allotment_ratio_pct | 1,235 | -1.0000 | -0.5223 |
| a_investor_lottery_rate_pct | 1,194 | -0.9612 | -0.8035 |
| offline_oversubscription_ratio_before_clawback | 1,235 | 0.9402 | 0.7818 |
| inquiry_oversubscription_ratio | 1,235 | 0.8004 | 0.7650 |
| offline_allotment_accounts | 1,235 | 0.7969 | 0.7792 |
| a_investor_allotted_accounts | 1,218 | 0.7617 | 0.7527 |
| offline_inquiry_investors | 1,235 | 0.6402 | 0.3668 |
| inquiry_allotment_accounts | 1,235 | 0.5880 | 0.7466 |
| offline_valid_quote_subscription_10k | 1,235 | 0.5166 | 0.1642 |
| offline_subscription_total_10k | 1,235 | 0.5166 | 0.1642 |
| inquiry_investors_count | 1,235 | 0.5053 | 0.1471 |
| excluded_subscription_share_pct | 1,235 | -0.4869 | -0.3393 |

## 6. 主要结论

- 板块差异很强：主板 网下超额认购倍数中位数最高 (8,622.80 倍)，北交所 最低 (21.37 倍)。
- 预测前可用字段中，`inquiry_oversubscription_ratio` 与标签的 Spearman 相关绝对值最高，为 0.800，方向为正相关。
- `网下申购配售比例` 与 `100 / 网下超额认购倍数` 基本互为倒数，二者差异的 95% 分位仅 0.000000 个百分点；建模时只能作为标签。

## 7. 字段缺失情况

| field | missing_count | missing_rate |
| --- | --- | --- |
| offer_price_upper_yuan | 1,643 | 1.0000 |
| offer_price_lower_yuan | 1,643 | 1.0000 |
| offer_price_range_pct | 1,643 | 1.0000 |
| offer_price_position_in_range | 1,643 | 1.0000 |
| other_issue_10k | 1,640 | 0.9982 |
| strategic_allocation_10k | 470 | 0.2861 |
| strategic_allocation_share_pct | 470 | 0.2861 |
| a_investor_lottery_rate_pct | 449 | 0.2733 |
| a_investor_allotted_shares_10k | 425 | 0.2587 |
| a_investor_subscription_shares_10k | 425 | 0.2587 |

## 8. 异常值检查

| field | p01 | p99 | below_p01_count | above_p99_count | min_security | min | max_security | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline_oversubscription_ratio | 13.6334 | 10,730.7212 | 13 | 13 | 一诺威 | 1.2300 | 埃泰克 | 13,987.6532 |
| offline_allotment_ratio_pct | 0.0093 | 7.3519 | 13 | 13 | 埃泰克 | 0.0071 | 一诺威 | 81.0437 |
| offer_price_yuan | 3.3868 | 158.7754 | 17 | 17 | 龙腾光电 | 1.2200 | 禾迈股份 | 557.8000 |
| issue_amount_100m_yuan | 0.8338 | 68.9849 | 17 | 17 | 永顺生物 | 0.3586 | 中芯国际 | 532.3019 |
| ipo_pe_diluted | 10.1770 | 347.3049 | 16 | 16 | 海博思创 | 6.1400 | 孚能科技 | 1,737.4900 |
| inquiry_subscription_total_10k | 24,489.3164 | 69,560,444.4000 | 13 | 13 | 永顺生物 | 1,452.8800 | 和辉光电-U | 227,025,750.0000 |
| inquiry_investors_count | 211.6800 | 843.2800 | 13 | 13 | 一诺威 | 58.0000 | 贝特瑞 | 1,175.0000 |

## 9. 建模建议（v2）

- 询价阶段字段（inquiry_subscription_total_10k、inquiry_investors_count 等）现已可用，
  是最有价值的新增特征，建议优先加入 Ridge 和 LightGBM 基线。
- inquiry_oversubscription_ratio（初步询价超额认购倍数）理论上是最强预测因子，
  但必须确认其发布时点确实早于网下申购截止。
- recent_ipo_first_day_return_ma20 已基于过去 20 只 IPO 首日涨幅滚动计算，可作为
  市场热度代理变量，不引入未来数据。
- 北交所标签覆盖率仅约 13%（41/316），短期内建议与其他板块合并训练并单独评估误差，
  不单独建模。
- 主板仍无询价补充字段，如需统一模型需补齐或使用缺失值插补策略。
