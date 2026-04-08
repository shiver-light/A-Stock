# 最小选股逻辑

本目录只负责把因子结果转换为股票选择结果，不处理收益、交易价格、手续费或调仓执行。

## 输出结构

- `trade_date`
- `ts_code`
- `score`
- `rank`
- `selected`

## 函数

- `combine_factor_scores`
  - 输入多因子宽表
  - 按日做截面百分位排名
  - 默认等权合成为 `score`

- `rank_signal`
  - 输入单因子长表或已有 `score` 列
  - 按交易日独立排序

- `top_n_selection`
  - 按每个交易日选择前 `N` 名

## 时间口径

- 每个交易日只使用当日可得因子值
- 默认用于 T 日收盘形成信号，供 T+1 执行
