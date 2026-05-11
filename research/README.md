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

配套验收标准见：

- [research/FACTOR_ACCEPTANCE.md](/Users/raymond/src/A-Stock/research/FACTOR_ACCEPTANCE.md)

## 运行入口

当前仓库已经提供最小 CLI 入口：

```bash
python3 -m research run --config research/experiments.yaml
```

当前也提供日度荐股共识入口：

```bash
python3 -m research recommend-consensus --run-dir <run_dir> --as-of-date YYYYMMDD
```

收盘后归档下一交易日的三类共识荐股：

```bash
python3 -m research archive-daily-consensus --signal-date YYYYMMDD
```

最常用参数：

- `--config`
  - 指定实验配置文件路径
- `--output-dir`
  - 指定实验输出根目录，默认 `research_runs`
- `--run-name`
  - 指定本次实验目录名；不传则自动生成
- `--core-models`
  - 指定主交易模型
- `--confirm-models`
  - 指定确认模型
- `--watch-models`
  - 指定观察模型；不传则默认为空
- `--archive-dir`
  - 指定日度荐股归档根目录；默认 `~/Documents/A-Stock/daily_recommendations`

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

## 共识荐股入口

`recommend-consensus` 用于把已完成的 research run 转成日度荐股结果。

输出分三层：

- `trade_consensus`
  - 同时命中主模型和确认模型的 `A` 级候选
- `trade_core`
  - 主模型命中的 `A + B` 级候选
- `watch_list`
  - 只在确认模型或观察模型里命中的 `C` 级候选

### HS300 示例

基于 [research_runs/hs300_turnover_momentum_stage2](/Users/raymond/src/A-Stock/research_runs/hs300_turnover_momentum_stage2)：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/hs300_turnover_momentum_stage2 \
  --as-of-date 20260501 \
  --core-models hstm2_04_hs300_turnover50_ret60_40_rev5_10_top10 \
  --confirm-models hstm2_02_hs300_turnover60_ret60_30_rev5_10_top15 \
  --watch-models hstm2_07_hs300_turnover40_ret60_40_rev5_20_top10
```

如果要 JSON：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/hs300_turnover_momentum_stage2 \
  --as-of-date 20260501 \
  --core-models hstm2_04_hs300_turnover50_ret60_40_rev5_10_top10 \
  --confirm-models hstm2_02_hs300_turnover60_ret60_30_rev5_10_top15 \
  --watch-models hstm2_07_hs300_turnover40_ret60_40_rev5_20_top10 \
  --output json
```

### ZZ500 示例

基于 [research_runs/zz500_value_stage3](/Users/raymond/src/A-Stock/research_runs/zz500_value_stage3)：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/zz500_value_stage3 \
  --as-of-date 20260501 \
  --core-models zz5v3_01_ep_ttm_top10 \
  --confirm-models zz5v3_11_ep50_bp20_ret60_30_top20 \
  --watch-models zz5v3_02_ep_ttm_top20
```

如果要 JSON：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/zz500_value_stage3 \
  --as-of-date 20260501 \
  --core-models zz5v3_01_ep_ttm_top10 \
  --confirm-models zz5v3_11_ep50_bp20_ret60_30_top20 \
  --watch-models zz5v3_02_ep_ttm_top20 \
  --output json
```

说明：

- `HS300` 当前建议：
  - 主模型 `hstm2_04_hs300_turnover50_ret60_40_rev5_10_top10`
  - 确认模型 `hstm2_02_hs300_turnover60_ret60_30_rev5_10_top15`
  - 观察模型 `hstm2_07_hs300_turnover40_ret60_40_rev5_20_top10`
- `ZZ500` 当前建议：
  - 主模型 `zz5v3_01_ep_ttm_top10`
  - 确认模型 `zz5v3_11_ep50_bp20_ret60_30_top20`
  - 观察模型 `zz5v3_02_ep_ttm_top20`
- 如果不传 `--watch-models`，当前不会自动补旧模型名。

### ZZ1000 观察示例

基于 [research_runs/zz1000_structure_stage4](/Users/raymond/src/A-Stock/research_runs/zz1000_structure_stage4)：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/zz1000_structure_stage4 \
  --as-of-date 20260501 \
  --core-models st4_01_zz1000_position_safety_top50 \
  --confirm-models st4_07_zz1000_turnover_stability_close_high_safety_top50
```

如果要 JSON：

```bash
python3 -m research recommend-consensus \
  --run-dir research_runs/zz1000_structure_stage4 \
  --as-of-date 20260501 \
  --core-models st4_01_zz1000_position_safety_top50 \
  --confirm-models st4_07_zz1000_turnover_stability_close_high_safety_top50 \
  --output json
```

说明：

- `ZZ1000` 当前建议是 `watch-only`，不直接进入正式交易主池。
- 主观察模型：
  - `st4_01_zz1000_position_safety_top50`
- 确认模型：
  - `st4_07_zz1000_turnover_stability_close_high_safety_top50`
- 推荐层级仍然是：
  - `A`: 主观察模型和确认模型同时命中
  - `B`: 仅主观察模型命中
  - `C`: 仅确认模型命中

## 收盘后自动归档

`archive-daily-consensus` 用于在交易日收盘后，用当日收盘数据生成下一交易日的共识荐股，并按下一交易日日期归档。

默认输出目录：

```text
~/Documents/A-Stock/daily_recommendations/<target_trade_date>/
```

每个日期目录包含：

- `hs300.json` / `hs300.txt`
- `zz500.json` / `zz500.txt`
- `zz1000.json` / `zz1000.txt`
- `manifest.json`
- `summary.txt`

手动运行：

```bash
python3 -m research archive-daily-consensus \
  --signal-date 20260507
```

指定归档目录：

```bash
python3 -m research archive-daily-consensus \
  --signal-date 20260507 \
  --archive-dir ~/Documents/A-Stock/daily_recommendations
```

适合放进本机定时任务，每个交易日 `15:05` 执行一次。若当天不是交易日，命令会跳过且不写归档。

仓库也提供了一个薄脚本：

```bash
scripts/archive_daily_consensus.sh
```

macOS 上可用 `crontab -e` 加一条本机定时任务：

```cron
5 15 * * 1-5 cd /Users/raymond/src/A-Stock && scripts/archive_daily_consensus.sh >> /Users/raymond/src/A-Stock/output/daily_consensus.log 2>&1
```

这条任务会在工作日 `15:05` 运行。遇到 A 股非交易日时，命令会按交易日历跳过。

如果要使用指定虚拟环境，并保存到访达的“文稿”目录下：

```cron
5 15 * * 1-5 cd /Users/raymond/src/A-Stock && source ~/src/aquant/venv/bin/activate && scripts/archive_daily_consensus.sh --archive-dir "$HOME/Documents/A-Stock/daily_recommendations" >> /Users/raymond/src/A-Stock/output/daily_consensus.log 2>&1
```

归档会按下一交易日生成日期文件夹，例如：

```text
~/Documents/A-Stock/daily_recommendations/20260508/
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
