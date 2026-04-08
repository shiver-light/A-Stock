# AGENTS.md

本仓库用于 A股量化策略研究、选股、回测和策略评估。

agent 在修改代码时必须遵守以下规则：

---

# 1. 研究严谨性优先

优先级：

1. 回测严谨性
2. 可复现性
3. 可解释性
4. 代码简洁性

禁止引入未来函数（look-ahead bias）。

禁止使用未来数据生成信号。

---

# 2. 不得静默改变关键假设

以下参数不得被自动修改：

- rebalance frequency（调仓频率）
- fee model（手续费）
- slippage（滑点）
- benchmark（基准）
- universe（股票池）
- execution price（成交价格口径）

如需修改，必须明确说明。

---

# 3. 时间对齐规则

必须明确：

signal time
execution time
holding period

默认：

T日收盘生成信号
T+1日成交

除非策略明确说明其他规则。

---

# 4. A股市场约束

若策略涉及交易可行性：

需考虑：

停牌
涨停无法买入
跌停无法卖出
ST过滤（如项目已有）

不要默认美股规则。

---

# 5. Python代码规则

优先使用：

pandas
numpy

避免：

复杂元编程
隐式副作用
隐藏状态

函数应职责单一。

---

# 6. commit规则

修改代码必须生成 commit。

格式：

feat:
fix:
refactor:
perf:
docs:
test:

示例：

fix: 修复 forward return 时间错位问题

refactor: 拆分因子计算与信号生成逻辑

---

# 7. agent行为约束

agent 不应：

擅自改变回测逻辑
擅自改变股票池
擅自改变调仓周期

agent 应：

提醒潜在未来函数问题
提醒时间错位问题
说明关键假设
