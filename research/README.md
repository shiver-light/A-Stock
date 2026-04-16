# Research 层

用于批量运行一组有明确金融含义的候选实验，并统一汇总结果。

## 能力

- 读取实验配置
- 逐个调用现有 `pipeline`
- 汇总关键指标
- 支持按 `sharpe` / `excess_cumulative_return` / `max_drawdown` 排序
- 支持每个实验独立落盘
- 支持中断后 resume
- 支持基于已完成实验生成 partial summary

## 设计原则

- 不做暴力调参
- 不以收益最高作为唯一目标
- 优先比较：
  - universe
  - top_n
  - factor 组合
  - factor 权重

## 输出目录结构

```text
research_runs/
  <run_name>/
    run_config.yaml
    run_status.json
    summary.parquet
    summary.csv
    experiments/
      <experiment_name>/
        config.json
        status.json
        metrics.json
        latest_selection.json
        report.json
        report.txt
        error.txt
```
