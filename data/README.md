# A股日线数据函数

统一日线数据函数：`get_a_share_daily_prices`

## 接口选择

- 使用接口：`daily`
- 文档：https://tushare.pro/document/2?doc_id=27
- 选择原因：
  - 已覆盖目标字段：`trade_date, ts_code, open, high, low, close, vol, amount`
  - 权限门槛更低，`120` 积分起
  - 官方说明基础积分每分钟约 `500` 次，单次 `6000` 行
  - 未复权底表更适合作为后续因子、模型、回测统一输入

## 函数签名

```python
from data import get_a_share_daily_prices

df = get_a_share_daily_prices(
    ts_code="000001.SZ",
    start_date="20240101",
    end_date="20240131",
    refresh=False,
)
```

## 返回字段

- `trade_date`
- `ts_code`
- `open`
- `high`
- `low`
- `close`
- `vol`
- `amount`

## 行为约定

- 返回 `pandas.DataFrame`
- 按 `trade_date`, `ts_code` 升序排序
- 字段名保持 Tushare 原始命名
- token 仅从环境变量读取：`TUSHARE_TOKEN` 或 `TS_TOKEN`
- 支持本地 parquet 缓存，默认目录：`data/cache/tushare/`
