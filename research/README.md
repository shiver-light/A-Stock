# Research 层

用于批量运行一组有明确金融含义的候选实验，并统一汇总结果。

## 能力

- 读取实验配置
- 逐个调用现有 `pipeline`
- 汇总关键指标
- 支持按 `sharpe` / `excess_cumulative_return` / `max_drawdown` 排序

## 设计原则

- 不做暴力调参
- 不以收益最高作为唯一目标
- 优先比较：
  - universe
  - top_n
  - factor 组合
  - factor 权重
