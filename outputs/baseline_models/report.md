# 三阶段模型回测报告

生成日期：2026-05-22

## 设计原则

| 时间节点 | 含义 | 可用信息 |
|---|---|---|
| T-6 | 申购决策期 | 招股书、询价公告（申购上下限/步长）、行业PE、历史热度 |
| T-1 | 回拨前预测 | T-6全部 + 询价结果（申购总量、机构数、价格分布、最终发行价）|
| T+1 | 回拨后预测 | T-1全部 + 回拨比例 |
| T+2 | 目标变量 | 网下超额认购倍数（禁止作为输入）|

演示模型 = **T-1 LightGBM**（不使用任何网下申购数据）

## OOS 整体指标

| model | stage | n | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| board_mean_t6 | T6 | 624 | 0.3704 | 0.5318 | 0.5293 | 0.2700 |
| lgbm_t1 | T1 | 624 | 0.1155 | 0.2412 | 0.9031 | 0.9531 |
| lgbm_t1plus | T1PLUS | 624 | 0.1079 | 0.2222 | 0.9178 | 0.9700 |
| lgbm_t6 | T6 | 624 | 0.3481 | 0.4830 | 0.6117 | 0.5117 |
| ridge_t1 | T1 | 624 | 0.3087 | 0.7252 | 0.1247 | 0.7364 |

> 最佳模型：**lgbm_t1plus** [T1PLUS]
> Spearman=0.9700 MAE=0.1079
>
> 演示模型（lgbm_t1）：Spearman=0.9531 MAE=0.1155

## 分板块 OOS 指标

| model | stage | board | n | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_mean_t6 | T6 | 主板 | 105 | 0.6634 | 0.8881 | -2.3505 | 0.1738 |
| board_mean_t6 | T6 | 创业板 | 284 | 0.2248 | 0.2903 | -0.1800 | -0.3903 |
| board_mean_t6 | T6 | 北交所 | 6 | 1.6860 | 1.8562 | -0.0463 | -0.9258 |
| board_mean_t6 | T6 | 科创板 | 229 | 0.3820 | 0.4628 | -0.1142 | 0.3011 |
| lgbm_t1 | T1 | 主板 | 105 | 0.3622 | 0.4900 | -0.0198 | 0.6397 |
| lgbm_t1 | T1 | 创业板 | 284 | 0.0513 | 0.0671 | 0.9369 | 0.9794 |
| lgbm_t1 | T1 | 北交所 | 6 | 0.8517 | 1.0673 | 0.6541 | 0.8857 |
| lgbm_t1 | T1 | 科创板 | 229 | 0.0626 | 0.1142 | 0.9321 | 0.9857 |
| lgbm_t1plus | T1PLUS | 主板 | 105 | 0.3086 | 0.4259 | 0.2294 | 0.6519 |
| lgbm_t1plus | T1PLUS | 创业板 | 284 | 0.0530 | 0.0722 | 0.9270 | 0.9797 |
| lgbm_t1plus | T1PLUS | 北交所 | 6 | 0.8889 | 1.0839 | 0.6433 | 0.8857 |
| lgbm_t1plus | T1PLUS | 科创板 | 229 | 0.0636 | 0.1190 | 0.9263 | 0.9858 |
| lgbm_t6 | T6 | 主板 | 105 | 0.6132 | 0.7626 | -1.4701 | 0.3123 |
| lgbm_t6 | T6 | 创业板 | 284 | 0.2547 | 0.3242 | -0.4718 | 0.2529 |
| lgbm_t6 | T6 | 北交所 | 6 | 1.6261 | 1.8451 | -0.0338 | 0.1429 |
| lgbm_t6 | T6 | 科创板 | 229 | 0.3088 | 0.3866 | 0.2224 | 0.4025 |
| ridge_t1 | T1 | 主板 | 105 | 0.8011 | 1.1325 | -4.4483 | 0.4866 |
| ridge_t1 | T1 | 创业板 | 284 | 0.1904 | 0.7560 | -7.0012 | 0.9242 |
| ridge_t1 | T1 | 北交所 | 6 | 1.5720 | 1.7417 | 0.0788 | 0.0286 |
| ridge_t1 | T1 | 科创板 | 229 | 0.1965 | 0.2381 | 0.7051 | 0.9284 |

## T-6 vs T-1 vs T+1 信息增益

| 阶段跃升 | Spearman 提升 | 含义 |
|---|---|---|
| T-6 → T-1 | 询价结果的价值 | 加入 inquiry_oversubscription_ratio 等 |
| T-1 → T+1 | 回拨比例的价值 | 加入 clawback_ratio_pct |

## 特征重要性 Top-10（lgbm_t1）

| feature | importance_gain | importance_split |
| --- | --- | --- |
| num__inquiry_oversubscription_ratio | 9279.5 | 951 |
| num__inquiry_allotment_accounts | 4881.5 | 335 |
| num__inquiry_investors_count | 1005.2 | 535 |
| num__excluded_subscription_share_pct | 445.1 | 687 |
| num__issue_amount_100m_yuan | 156.3 | 237 |
| num__subscription_step_10k | 76.0 | 1 |
| cat__board_主板 | 68.0 | 50 |
| num__inquiry_subscription_total_10k | 46.3 | 173 |
| cat__board_科创板 | 42.5 | 174 |
| num__high_price_excluded_subscription_share_pct | 40.8 | 142 |

## 保存的模型文件

所有模型已用 joblib 序列化至 `outputs/baseline_models/models/`。
使用 `scripts/predict.py` 加载并预测新 IPO。
