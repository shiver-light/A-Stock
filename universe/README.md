# Universe 模块

统一定义某个 `as_of_date` 可选哪些股票。

也支持按一组历史日期批量获取：

- `get_universe_history(universe_name, as_of_dates, ...)`

## 支持的 universe

- `all_a`
- `hs300`
- `zz500`
- `zz1000`
- `sse50`
- `main_board`
- `chinext`
- `custom`

## 历史一致性说明

- `hs300` / `zz500` / `zz1000` / `sse50`
  - 使用 `index_weight`
  - 取 `trade_date <= as_of_date` 的最近一期成分
  - 属于月度历史成分口径

- `all_a`
  - 使用 `stock_basic + list_date + delist_date`
  - 避免只用当前上市股票列表回填历史

- `main_board` / `chinext`
  - 基于 `stock_basic.market` 字段做历史近似
  - 未追踪极少数历史板块迁移事件

- `custom`
  - 使用外部传入 `ts_codes`

## 统一输出字段

- `as_of_date`
- `ts_code`
- `universe_name`
- `in_universe`

指数成分型 universe 可额外包含：

- `weight`
