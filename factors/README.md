# 基础因子库

## 时间口径

- 技术因子使用前复权日线收盘价
- 财务因子使用 `ann_date` 的下一交易日开始生效
- 因子函数只输出：
  - `trade_date`
  - `ts_code`
  - `factor_name`
  - `factor_value`

## 因子定义

- `return_5d`: 5日收益率
- `return_20d`: 20日收益率
- `volatility_20d`: 20日日收益率标准差
- `turnover_mean_20d`: 20日平均自由流通换手率
- `price_rank_60d`: 60日价格区间分位 `(close - rolling_min) / (rolling_max - rolling_min)`
- `pe_ttm`: TTM市盈率
- `pb`: 市净率
- `roe`: 净资产收益率
- `revenue_growth`: 营业收入同比增速，使用 `or_yoy`

## 数据来源

- 日线行情：`daily` + `adj_factor`
- 日度估值：`daily_basic`
- 财务指标：`fina_indicator`
- 交易日历：`trade_cal`
