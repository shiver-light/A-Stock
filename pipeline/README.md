# 最小量化流程入口

流程：

数据获取
→ 因子计算
→ 选股信号生成
→ 回测
→ 最新推荐结果输出
→ 策略报告输出

## 主入口

`run_minimal_pipeline`

输入：

- `ts_codes`
- `start_date`
- `end_date`
- `top_n`

输出：

- `factor_data`
- `scored_signals`
- `selected_signals`
- `strategy_returns`
- `holdings`
- `performance`
- `latest_selection`
- `report`
- `report_text`

## 命令行执行

按 universe 运行：

```bash
export TUSHARE_TOKEN=your_token
python3 -m pipeline \
  --universe-name hs300 \
  --start-date 20250101 \
  --end-date 20250331 \
  --top-n 5 \
  --benchmark-code 000300.SH
```

按自定义股票列表运行：

```bash
export TUSHARE_TOKEN=your_token
python3 -m pipeline \
  --ts-codes 000001.SZ 000002.SZ 600000.SH \
  --start-date 20250101 \
  --end-date 20250331 \
  --top-n 3
```

如果需要结构化输出：

```bash
python3 -m pipeline \
  --universe-name hs300 \
  --start-date 20250101 \
  --end-date 20250331 \
  --output json
```
