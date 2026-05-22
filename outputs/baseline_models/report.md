# 基准模型回测报告

生成日期：2026-05-22

## 回测设计

- 目标变量：`log_offline_oversubscription` = log(网下超额认购倍数)
- 排序依据：`subscription_deadline_date`（申购截止日），缺失时回退到 `listing_date`
- 回测方式：扩展窗口（expanding window），每折用截止日之前全部数据训练
- 截止日期：2022-01-01 / 2023-01-01 / 2024-01-01 / 2025-01-01
- OOS 样本总数：624 只不同 IPO

## 模型

| 模型 | 特征集 | 说明 |
|---|---|---|
| board_mean | — | 训练集中同板块均值（无模型基线）|
| board_recent_mean | — | 同板块最近 20 条样本滚动均值 |
| ridge_basic | FEATS_BASIC | Ridge，无询价字段 |
| ridge_inquiry | FEATS_INQUIRY | Ridge，含询价字段 |
| lgbm_inquiry | FEATS_INQUIRY | LightGBM，含询价字段 |

## 整体 OOS 指标

| model | n | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- |
| board_mean | 624 | 0.3704 | 0.5318 | 0.5293 | 0.2700 |
| board_recent_mean | 624 | 0.3472 | 0.5196 | 0.5506 | 0.2996 |
| lgbm_inquiry | 624 | 0.1130 | 0.2398 | 0.9043 | 0.9536 |
| ridge_basic | 624 | 0.3877 | 0.6809 | 0.2284 | 0.5396 |
| ridge_inquiry | 624 | 0.3196 | 0.6993 | 0.1860 | 0.7551 |

> 最佳模型（Spearman）：**lgbm_inquiry**，Spearman=0.9536，MAE=0.1130

## 分板块 OOS 指标

| model | board | n | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| board_mean | 主板 | 105 | 0.6634 | 0.8881 | -2.3505 | 0.1738 |
| board_mean | 创业板 | 284 | 0.2248 | 0.2903 | -0.1800 | -0.3903 |
| board_mean | 北交所 | 6 | 1.6860 | 1.8562 | -0.0463 | -0.9258 |
| board_mean | 科创板 | 229 | 0.3820 | 0.4628 | -0.1142 | 0.3011 |
| board_recent_mean | 主板 | 105 | 0.6432 | 0.8836 | -2.3166 | 0.1738 |
| board_recent_mean | 创业板 | 284 | 0.2312 | 0.2908 | -0.1837 | -0.0598 |
| board_recent_mean | 北交所 | 6 | 1.6841 | 1.8540 | -0.0438 | 0.0000 |
| board_recent_mean | 科创板 | 229 | 0.3202 | 0.4276 | 0.0487 | 0.2576 |
| lgbm_inquiry | 主板 | 105 | 0.3616 | 0.4932 | -0.0333 | 0.6316 |
| lgbm_inquiry | 创业板 | 284 | 0.0515 | 0.0696 | 0.9322 | 0.9795 |
| lgbm_inquiry | 北交所 | 6 | 0.7738 | 1.0081 | 0.6914 | 0.8857 |
| lgbm_inquiry | 科创板 | 229 | 0.0580 | 0.1121 | 0.9346 | 0.9876 |
| ridge_basic | 主板 | 105 | 0.9043 | 1.3729 | -7.0068 | 0.3686 |
| ridge_basic | 创业板 | 284 | 0.2126 | 0.2736 | -0.0482 | 0.6574 |
| ridge_basic | 北交所 | 6 | 1.9824 | 2.2860 | -0.5869 | -0.7714 |
| ridge_basic | 科创板 | 229 | 0.3262 | 0.4113 | 0.1199 | 0.6206 |
| ridge_inquiry | 主板 | 105 | 0.7551 | 1.0840 | -3.9914 | 0.5351 |
| ridge_inquiry | 创业板 | 284 | 0.1873 | 0.7079 | -6.0156 | 0.9493 |
| ridge_inquiry | 北交所 | 6 | 1.7596 | 1.9047 | -0.1016 | -0.0286 |
| ridge_inquiry | 科创板 | 229 | 0.2464 | 0.2783 | 0.5972 | 0.9603 |

## 分年度 OOS 指标

| model | listing_year | n | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| board_mean | 2022.0000 | 242 | 0.3211 | 0.4138 | 0.5923 | 0.3072 |
| board_mean | 2023.0000 | 204 | 0.4860 | 0.7069 | 0.3726 | -0.3265 |
| board_mean | 2024.0000 | 66 | 0.2280 | 0.3244 | 0.6343 | 0.7764 |
| board_mean | 2025.0000 | 83 | 0.2750 | 0.3746 | 0.4996 | 0.8297 |
| board_mean | 2026.0000 | 29 | 0.5651 | 0.7077 | 0.1162 | 0.9242 |
| board_recent_mean | 2022.0000 | 242 | 0.3062 | 0.4079 | 0.6039 | 0.3072 |
| board_recent_mean | 2023.0000 | 204 | 0.4626 | 0.6960 | 0.3917 | -0.4114 |
| board_recent_mean | 2024.0000 | 66 | 0.2140 | 0.3279 | 0.6262 | 0.7673 |
| board_recent_mean | 2025.0000 | 83 | 0.2440 | 0.3474 | 0.5696 | 0.8128 |
| board_recent_mean | 2026.0000 | 29 | 0.4758 | 0.6510 | 0.2521 | 0.9242 |
| lgbm_inquiry | 2022.0000 | 242 | 0.0651 | 0.1297 | 0.9599 | 0.9844 |
| lgbm_inquiry | 2023.0000 | 204 | 0.1785 | 0.3668 | 0.8311 | 0.8919 |
| lgbm_inquiry | 2024.0000 | 66 | 0.0962 | 0.1808 | 0.8864 | 0.9324 |
| lgbm_inquiry | 2025.0000 | 83 | 0.0831 | 0.1138 | 0.9538 | 0.9729 |
| lgbm_inquiry | 2026.0000 | 29 | 0.1764 | 0.1987 | 0.9303 | 0.9818 |
| ridge_basic | 2022.0000 | 242 | 0.3094 | 0.4084 | 0.6029 | 0.5689 |
| ridge_basic | 2023.0000 | 204 | 0.5884 | 1.0433 | -0.3668 | 0.0531 |
| ridge_basic | 2024.0000 | 66 | 0.3415 | 0.4804 | 0.1977 | 0.7940 |
| ridge_basic | 2025.0000 | 83 | 0.1682 | 0.2248 | 0.8198 | 0.9289 |
| ridge_basic | 2026.0000 | 29 | 0.3627 | 0.5058 | 0.5484 | 0.9350 |
| ridge_inquiry | 2022.0000 | 242 | 0.2869 | 0.7986 | -0.5183 | 0.9440 |
| ridge_inquiry | 2023.0000 | 204 | 0.4388 | 0.8048 | 0.1866 | 0.2757 |
| ridge_inquiry | 2024.0000 | 66 | 0.2807 | 0.3623 | 0.5438 | 0.9223 |
| ridge_inquiry | 2025.0000 | 83 | 0.1976 | 0.2392 | 0.7959 | 0.9747 |
| ridge_inquiry | 2026.0000 | 29 | 0.1929 | 0.4273 | 0.6777 | 0.9857 |

## LightGBM 特征重要性（Top 15，全量训练）

| feature | importance_gain | importance_split |
| --- | --- | --- |
| num__inquiry_oversubscription_ratio | 10346.4 | 947 |
| num__inquiry_allotment_accounts | 3655.3 | 338 |
| num__inquiry_investors_count | 916.5 | 437 |
| num__excluded_subscription_share_pct | 462.7 | 706 |
| num__subscription_step_10k | 318.5 | 1 |
| num__issue_amount_100m_yuan | 156.4 | 225 |
| cat__board_主板 | 141.4 | 47 |
| num__inquiry_subscription_total_10k | 100.1 | 173 |
| num__high_price_excluded_subscription_share_pct | 37.6 | 132 |
| cat__board_科创板 | 31.3 | 149 |
| num__quote_price_vs_offer | 25.9 | 203 |
| num__offline_issue_before_share_pct | 25.5 | 343 |
| num__subscription_upper_limit_10k | 25.4 | 100 |
| cat__board_创业板 | 19.7 | 72 |
| num__recent_ipo_first_day_return_ma20 | 18.5 | 184 |

## 解读要点

- **Spearman 排名相关**是最重要指标，直接反映"哪只新股更值得优先申购"的排序能力。
- `inquiry_oversubscription_ratio`（初步询价超额认购倍数）预计是最强预测因子；如果
  特征重要性显示其他字段更重要，需要重新审视数据质量。
- 北交所由于分布极端（中位超额认购仅 21 倍），单独看其误差具有参考价值。
- Ridge 与 LightGBM 的差距可以量化询价字段对线性 vs 非线性关系的贡献。
