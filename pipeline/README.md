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
