# 最小日频回测模块

## 假设

- 调仓频率：每月第一个交易日
- 信号时间：前一个交易日收盘
- 执行时间：调仓日开盘
- 权重：等权
- 手续费：单边 10 bps
- 滑点：未单独建模

## 时间顺序

- `signal_date = T`
- `execution_date = T+1`
- 调仓日收益使用 `close / open - 1`
- 非调仓日收益使用 `close_t / close_{t-1} - 1`

## 输出

- 策略收益序列：
  - `trade_date`
  - `strategy_return`
- 持仓明细：
  - `trade_date`
  - `ts_code`
  - `weight`
