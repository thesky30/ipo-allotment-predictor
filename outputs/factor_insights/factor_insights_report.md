# 询价前因子洞察与板块时期特征报告

生成日期：2026-05-28

## 领导速览

- 主板网下超额认购倍数中位数最高（8623倍），约为北交所（21倍）的403.5倍，板块制度差异是第一层结构。
- 市场成交额 `market_turnover_ma20` 在T-6模型平均绝对SHAP中排名第2，平均贡献约0.158（log空间），是关键外部环境因子。
- 单因子IC最高的是 `recent_ipo_first_day_return_ma20`（Spearman IC=0.374），可作为询价前重点跟踪因子。
- 市场成交额从低分位到高分位时，实际中签率中位数由0.0492%变化到0.0250%，可用于向领导解释流动性环境分层。
- 2024-2026 当前阶段的市场成交额中位数最高，说明不同年份/时期的资金环境不可混为一谈。
- T-1/T+1仅作为信息增益上界；本报告所有核心因子均按T-6询价前口径输出。

## 板块画像

| board | sample_count | label_count | median_oversubscription | median_subscription_rate_pct | median_first_day_return_pct | break_rate | median_total_issue_shares_10k | median_offline_issue_before_10k | median_offline_issue_before_share_pct | median_strategic_allocation_share_pct | median_subscription_upper_limit_10k | median_subscription_lower_limit_10k | median_market_turnover_ma20 | median_recent_ipo_return_ma20 | median_concurrent_ipo_count | median_same_board_break_rate_ma10 | label_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 主板 | 111.00 | 105.00 | 8622.80 | 0.01 | 112.79 | 0.01 | 4490.86 | 2596.17 | 60.00 | 17.77 | 1290.00 | 100.00 | 11328.50 | 177.57 | 4.00 | 0.00 | 0.95 |
| 创业板 | 602.00 | 481.00 | 3367.41 | 0.03 | 93.43 | 0.09 | 2640.00 | 2145.00 | 71.50 | 10.00 | 1000.00 | 100.00 | 9793.90 | 163.69 | 12.00 | 0.00 | 0.80 |
| 北交所 | 316.00 | 41.00 | 21.37 | 4.68 | 22.06 | 0.26 | 1799.98 | 1300.00 | 58.69 | 17.39 | 1043.52 | 5.00 | 9468.76 | 26.21 | 24.00 | 0.20 | 0.13 |
| 科创板 | 610.00 | 608.00 | 2596.93 | 0.04 | 91.98 | 0.12 | 3339.00 | 2162.64 | 66.50 | 10.92 | 915.00 | 100.00 | 9250.81 | 131.20 | 12.00 | 0.00 | 1.00 |

## 时期画像

| period | sample_count | label_count | median_oversubscription | median_subscription_rate_pct | median_first_day_return_pct | break_rate | median_total_issue_shares_10k | median_offline_issue_before_10k | median_offline_issue_before_share_pct | median_strategic_allocation_share_pct | median_subscription_upper_limit_10k | median_subscription_lower_limit_10k | median_market_turnover_ma20 | median_recent_ipo_return_ma20 | median_concurrent_ipo_count | median_same_board_break_rate_ma10 | label_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-2020 注册制早期 | 314.00 | 296.00 | 1738.21 | 0.06 | 120.24 | 0.06 | 3000.00 | 2022.47 | 66.50 | 5.00 | 800.00 | 100.00 | 7888.40 | 158.38 | 14.00 | 0.00 | 0.94 |
| 2021-2022 扩容与破发阶段 | 758.00 | 557.00 | 3379.18 | 0.03 | 56.61 | 0.19 | 2560.60 | 2145.00 | 67.42 | 14.24 | 1000.00 | 100.00 | 9872.97 | 67.79 | 13.00 | 0.10 | 0.73 |
| 2023 全面注册制切换 | 290.00 | 204.00 | 3251.08 | 0.03 | 35.83 | 0.18 | 2622.54 | 2373.21 | 68.69 | 17.39 | 1100.00 | 100.00 | 8910.94 | 36.96 | 11.00 | 0.20 | 0.70 |
| 2024-2026 当前阶段 | 273.00 | 178.00 | 4249.40 | 0.02 | 185.15 | 0.00 | 3000.00 | 2448.58 | 60.00 | 17.39 | 1200.00 | 100.00 | 17377.41 | 213.24 | 3.00 | 0.00 | 0.65 |
| 未知时期 | 4.00 | 0.00 |  |  |  | 0.00 |  |  |  |  |  |  |  | 224.02 |  |  | 0.00 |

## T-6 单因子 IC Top

IC 是因子与未来 `log(网下超额认购倍数)` 的 Spearman 排序相关。正值表示因子越高，未来超额认购通常越高、中签率通常越低。

| factor | factor_group | n | spearman_ic | icir_by_year |
| --- | --- | --- | --- | --- |
| recent_ipo_first_day_return_ma20 | 市场情绪因子 | 1215.000 | 0.374 | 0.162 |
| subscription_step_10k | 申购规则因子 | 1235.000 | 0.305 | 1.428 |
| same_board_break_rate_ma10 | 市场情绪因子 | 1153.000 | -0.301 | -0.567 |
| market_turnover_ma20 | 市场流动性因子 | 1235.000 | 0.294 | -0.611 |
| same_board_concurrent_ipo_count | IPO供给拥挤因子 | 1235.000 | -0.240 | -0.214 |
| subscription_lower_limit_10k | 申购规则因子 | 1235.000 | 0.228 | 0.989 |
| comparable_pe_avg_ex_nonrecurring | 估值因子 | 1124.000 | -0.204 | -0.624 |
| strategic_allocation_share_pct | 发行供给因子 | 900.000 | 0.176 | -0.158 |
| concurrent_ipo_count | IPO供给拥挤因子 | 1235.000 | -0.168 | -0.493 |
| offline_issue_before_share_pct | 发行供给因子 | 1235.000 | -0.161 | -0.875 |
| concurrent_offline_issue_sum_10k | IPO供给拥挤因子 | 1227.000 | -0.084 | -0.097 |
| market_turnover_pct_rank_1y | 市场流动性因子 | 1235.000 | 0.083 | -0.525 |

## T-6 模型解释贡献 Top

以下为 LightGBM TreeSHAP 平均绝对贡献，含义是“对模型预测结果的解释贡献”，不是严格因果检验。

| feature | n | mean_shap | mean_abs_shap |
| --- | --- | --- | --- |
| subscription_step_10k | 1235.000 | 0.005 | 0.198 |
| market_turnover_ma20 | 1235.000 | -0.003 | 0.158 |
| board | 1235.000 | 0.001 | 0.127 |
| recent_ipo_first_day_return_ma20 | 1235.000 | 0.005 | 0.093 |
| subscription_upper_limit_10k | 1235.000 | 0.010 | 0.089 |
| subscription_lower_limit_10k | 1235.000 | -0.003 | 0.086 |
| offline_issue_before_clawback_10k | 1235.000 | -0.001 | 0.073 |
| offline_issue_before_share_pct | 1235.000 | -0.007 | 0.063 |
| market_turnover_ma20_over_ma60 | 1235.000 | 0.005 | 0.051 |
| total_issue_shares_10k | 1235.000 | -0.005 | 0.043 |
| online_issue_before_clawback_10k | 1235.000 | -0.005 | 0.036 |
| same_board_concurrent_ipo_count | 1235.000 | 0.001 | 0.036 |

## 图表索引

- `figures/board_period_heatmap.svg`：板块 × 时期的超额认购倍数热力图
- `figures/board_factor_profile.svg`：各板块核心询价前因子画像
- `figures/factor_ic_bar_t6.svg`：T-6 因子 IC 排名
- `figures/market_turnover_bucket.svg`：市场成交额分组 vs 实际中签率
- `figures/factor_shap_by_board.svg`：各板块 Top 因子 SHAP 热力图
- `figures/factor_shap_by_year.svg`：因子贡献随年份变化
