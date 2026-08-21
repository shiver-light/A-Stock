# Universe 模块

统一定义某个 `as_of_date` 可选哪些股票。

也支持按一组历史日期批量获取：

- `get_universe_history(universe_name, as_of_dates, ...)`

## 支持的 universe

- `all_a`
- `all_a_ex_chinext_st`
- `all_a_ex_chinext_star_st`
- `hs300`
- `zz500`
- `zz1000`
- `zz2000`
- `zz2000_ex_bj`
- `sse50`
- `main_board`
- `chinext`
- `custom`

## 历史一致性说明

- `hs300` / `zz500` / `zz1000` / `zz2000` / `zz2000_ex_bj` / `sse50`
  - 使用 `index_weight`
  - 取 `trade_date <= as_of_date` 的最近一期成分
  - 属于月度历史成分口径
  - `zz2000_ex_bj` 会在 `zz2000` 成分上额外排除 `.BJ` 标的
  - 这是当前数据层对 `adj_factor` / `daily_basic` 尚未完整覆盖北交所时的近似研究口径

- `all_a`
  - 使用 `stock_basic + list_date + delist_date`
  - 避免只用当前上市股票列表回填历史

- `all_a_ex_chinext_st`
  - 在 `all_a` 基础上排除创业板与名称包含 `ST` 的股票
  - 不排除科创板，保留旧研究口径兼容性

- `all_a_ex_chinext_star_st`
  - 在 `all_a` 基础上排除创业板、科创板与名称包含 `ST` 的股票
  - 用于不希望纳入 300/301 创业板和 688 科创板的研究

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
