# 回测规则

本目录负责策略收益模拟。

---

# 1. 信号与交易分离

信号生成：

决定买什么

回测执行：

决定何时成交

必须分离。

---

# 2. 时间顺序

必须满足：

signal_time < execution_time

默认：

T收盘生成信号
T+1开盘成交

不得使用：

未来价格成交。

---

# 3. 收益计算

收益计算应基于：

weights.shift(1)

示例：

returns =
weights.shift(1) * price.pct_change()

避免：

使用当期权重乘当期收益。

---

# 4. 交易成本

必须显式包含：

fee
slippage

示例：

cost =
turnover * fee_bps / 10000

不得忽略成本。

---

# 5. 调仓逻辑

需明确：

调仓频率
调仓日期
持仓数量
权重归一化方式

避免：

隐式调仓。

---

# 6. benchmark一致性

benchmark收益计算需与策略收益对齐：

相同时间频率
相同价格类型

---

# 7. 不得修改关键假设

不得自动改变：

fee
slippage
rebalance frequency
benchmark
execution price
