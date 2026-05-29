# 上市公司因子数据处理报告

生成文件：`data/processed/company_factor_data.csv`

## 总览

- 清洗后有效行数：1323
- 与现有 IPO 样本库匹配：1320
- 现有 IPO 样本暂缺本因子表：316
- 因子表有、IPO 样本库没有：3

## 关键字段覆盖

| 字段 | 非空数 | 覆盖率 |
|---|---:|---:|
| `inquiry_announcement_date` | 1197 | 90.5% |
| `inquiry_start_date` | 1197 | 90.5% |
| `sw_level1_industry_code` | 1323 | 100.0% |
| `sw_level2_industry_code` | 1323 | 100.0% |
| `offline_market_value_threshold_10k_yuan` | 1197 | 90.5% |
| `expected_fundraising_100m_yuan` | 1320 | 99.8% |
| `lead_underwriter` | 1323 | 100.0% |
| `sponsor` | 1323 | 100.0% |
| `revenue_cagr_3y_pct` | 359 | 27.1% |
| `latest_revenue_100m_yuan` | 1314 | 99.3% |

## 分板块匹配情况

| board | ipo_rows | matched_factor | inquiry_start_non_null | sw_l1_non_null | threshold_non_null | underwriter_non_null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 主板 | 111 | 110 | 105 | 110 | 105 | 110 |
| 创业板 | 602 | 601 | 481 | 601 | 481 | 601 |
| 北交所 | 316 | 0 | 0 | 0 | 0 | 0 |
| 科创板 | 610 | 609 | 608 | 609 | 608 | 609 |

## 口径提示

- 本脚本只新增独立表 `company_factor_data` 和联表视图式产物 `ipo_company_factor_joined`，不覆盖 `ipo_offline_sample` 主表。
- `offer_price_upper_yuan_factor` / `offer_price_lower_yuan_factor` 在本次文件中全为空，暂不能形成价格区间因子。
- 当前文件给的是申万行业代码，不是行业名称；后续如要展示友好名称，需要再补申万行业代码-名称映射。
- 这些字段属于询价前因子候选，但仍建议在入模前逐项确认公告发布时间口径。
