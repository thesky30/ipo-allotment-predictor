# 新增询价前候选因子 IC 报告

## 样本

- 因子面板行数：1639
- 有训练标签行数：1235
- 使用 `inquiry_start_date` 作为预测日的行数：1194
- 目标变量：`log_offline_oversubscription`

## 全样本 IC Top 15

IC 为候选因子与未来 `log(网下超额认购倍数)` 的 Spearman 排序相关。
正 IC 表示因子越高，未来超额认购通常越高，中签率通常越低。

| factor | factor_group | n | spearman_ic | note |
| --- | --- | --- | --- | --- |
| issue_pb_factor | 估值 | 1191.000 | -0.463 | 发行市净率，需确认是否依赖最终发行价 |
| sw_l1_prior_log_oversub_mean | 行业历史热度 | 1168.000 | 0.436 | 同申万一级代码历史平均log网下超额认购 |
| underwriter_prior_log_oversub_mean | 承销商声誉 | 1118.000 | 0.389 | 主承销商历史平均log网下超额认购 |
| board_turnover_ma20 | 板块流动性 | 1114.000 | 0.382 | 同板块成交额20日均值，严格取询价开始日前一交易日 |
| revenue_cagr_3y_pct | 成长 | 355.000 | -0.340 | 近三年营收复合增长率，覆盖率较低 |
| expected_fundraising_100m_yuan | 发行规模 | 1194.000 | -0.256 | 首发预计募集资金，询价前候选 |
| log_expected_fundraising | 发行规模 | 1194.000 | -0.256 | log1p(预计募资额) |
| sw_l1_prior_ipo_count | 行业历史热度 | 1235.000 | 0.249 | 同申万一级代码历史IPO数量 |
| underwriter_prior_ipo_count | 承销商声誉 | 1235.000 | 0.230 | 主承销商历史已发生IPO数量 |
| comparable_pe_avg_ex_nonrecurring_factor | 估值 | 1122.000 | -0.206 | 可比上市公司PE均值，询价前候选 |
| sw_l1_prior_first_day_return_mean | 行业历史热度 | 1168.000 | 0.158 | 同申万一级代码历史平均首日涨幅 |
| underwriter_prior_first_day_return_mean | 承销商声誉 | 1118.000 | 0.112 | 主承销商历史平均首日涨幅 |
| sw_l1_prior_break_rate | 行业历史热度 | 1168.000 | 0.111 | 同申万一级代码历史破发率 |
| board_turnover_pct_rank_1y | 板块流动性 | 1095.000 | -0.072 | 同板块成交额20日均值一年分位 |
| industry_pe_at_ipo_factor | 估值 | 1174.000 | -0.056 | 首发时所属行业市盈率，询价前候选 |

## 因子覆盖率

| factor | factor_group | non_null | coverage |
| --- | --- | --- | --- |
| sw_l1_prior_ipo_count | 行业历史热度 | 1639.000 | 1.000 |
| underwriter_prior_ipo_count | 承销商声誉 | 1639.000 | 1.000 |
| issue_pb_factor | 估值 | 1317.000 | 0.804 |
| log_expected_fundraising | 发行规模 | 1317.000 | 0.804 |
| expected_fundraising_100m_yuan | 发行规模 | 1317.000 | 0.804 |
| log_latest_revenue | 公司规模 | 1311.000 | 0.800 |
| latest_revenue_100m_yuan | 公司规模 | 1311.000 | 0.800 |
| industry_pe_at_ipo_factor | 估值 | 1300.000 | 0.793 |
| sw_l1_prior_log_oversub_mean | 行业历史热度 | 1293.000 | 0.789 |
| sw_l1_prior_first_day_return_mean | 行业历史热度 | 1293.000 | 0.789 |
| sw_l1_prior_break_rate | 行业历史热度 | 1293.000 | 0.789 |
| comparable_pe_avg_ex_nonrecurring_factor | 估值 | 1241.000 | 0.757 |
| board_turnover_ma20 | 板块流动性 | 1240.000 | 0.757 |
| board_return_ma20 | 板块情绪 | 1240.000 | 0.757 |
| underwriter_prior_first_day_return_mean | 承销商声誉 | 1237.000 | 0.755 |
| underwriter_prior_log_oversub_mean | 承销商声誉 | 1237.000 | 0.755 |
| underwriter_prior_break_rate | 承销商声誉 | 1237.000 | 0.755 |
| board_turnover_ma20_over_ma60 | 板块流动性 | 1230.000 | 0.750 |
| board_turnover_pct_rank_1y | 板块流动性 | 1221.000 | 0.745 |
| offline_market_value_threshold_10k_yuan | 申购规则 | 1194.000 | 0.728 |
| revenue_cagr_3y_pct | 成长 | 357.000 | 0.218 |

## 重要限制

- 北交所缺板块行情，因此 `board_*` 因子在北交所为空。
- 申万一级行情表是行业名称，上市公司因子表是行业代码；缺少代码-名称映射，所以暂未生成行业行情滚动因子。
- `issue_pb_factor`、`industry_pe_at_ipo_factor` 等估值字段虽然已列入候选，但入模前仍需确认是否在询价开始前已经公开。
- 本报告是单因子筛选，不代表多因子模型最终权重；下一步应把通过筛选的因子加入滚动回测模型比较。
