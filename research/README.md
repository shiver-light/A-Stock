# Research 层

用于批量运行一组有明确金融含义的候选实验，并统一汇总结果。

## 最小运行前提

1. 安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

2. 配置 Tushare token，二选一：

```bash
export TUSHARE_TOKEN=你的token
```

或在本地创建 `config/settings.yaml`：

```yaml
# local only
tushare_token: "你的token"
```

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

## 运行入口

当前仓库已经提供最小 CLI 入口：

```bash
python3 -m research run --config research/experiments.yaml
```

最常用参数：

- `--config`
  - 指定实验配置文件路径
- `--output-dir`
  - 指定实验输出根目录，默认 `research_runs`
- `--run-name`
  - 指定本次实验目录名；不传则自动生成

## 推荐运行命令

运行第一批实验：

```bash
python3 -m research run \
  --config research/experiments.yaml \
  --output-dir research_runs \
  --run-name first_batch
```

中断后恢复：

```bash
python3 -m research resume \
  --config research/experiments.yaml \
  --output-dir research_runs \
  --run-name first_batch
```

查看当前已完成实验的阶段性汇总：

```bash
python3 -m research summary \
  --run-dir research_runs/first_batch \
  --sort-by sharpe
```

如果要看 JSON：

```bash
python3 -m research summary \
  --run-dir research_runs/first_batch \
  --sort-by excess_cumulative_return \
  --output json
```

## 配置文件

默认配置文件是 [research/experiments.yaml](/Users/raymond/src/A-Stock/research/experiments.yaml)。

最小结构：

```yaml
global:
  start_date: "20230101"
  end_date: "20260331"
  benchmark_code: "000300.SH"
  top_n: 20

experiments:
  - name: "r01_hs300_baseline_top10"
    universe_name: "hs300"
    top_n: 10
    factor_config:
      return_20d: 1.0
      volatility_20d: -1.0
      turnover_mean_20d: 1.0
```

当前已支持并稳定使用的配置字段：

- `start_date`
- `end_date`
- `benchmark_code`
- `top_n`
- `universe_name`
- `factor_config`
- `enable_factor_diagnostics`
- `analysis_horizons`
- `ts_codes`

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
        factor_diagnostics.json
        factor_report.txt
        error.txt
```

文件说明：

- `run_config.yaml`
  - 本次批量实验原始配置
- `run_status.json`
  - run 级状态
- `summary.parquet` / `summary.csv`
  - 当前已完成实验的阶段性汇总
- `experiments/<experiment_name>/metrics.json`
  - 单个实验核心回测指标
- `experiments/<experiment_name>/report.txt`
  - 单个实验策略文本报告
- `experiments/<experiment_name>/factor_diagnostics.json`
  - 单因子诊断结果
- `experiments/<experiment_name>/factor_report.txt`
  - 单因子诊断文本报告
- `experiments/<experiment_name>/error.txt`
  - 实验失败时的错误信息

## 如何开始第一批实验

建议先跑默认研究矩阵，不要一上来自己扩很多参数：

```bash
python3 -m research run \
  --config research/experiments.yaml \
  --output-dir research_runs \
  --run-name first_batch
```

跑完后优先看：

1. `research_runs/first_batch/summary.csv`
2. `python3 -m research summary --run-dir research_runs/first_batch --sort-by sharpe`
3. 单个实验目录里的 `metrics.json` / `report.txt` / `factor_diagnostics.json`

如果结果中断，不需要重新开始，直接执行 `resume`。
