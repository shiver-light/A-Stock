# Tushare A股日线接口封装

本目录提供统一的 Tushare Pro 访问层，避免在 `strategy/`、`factor/`、`backtest/` 中散落直接调用 `pro.xxx(...)`。

## 已封装接口

### `daily`
- 用途：A股未复权日线行情底表
- 文档：https://tushare.pro/document/2?doc_id=27
- 权限：120 积分起
- 关键字段：`ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount`
- 单次限制：文档说明单次最多 6000 行

### `adj_factor`
- 用途：A股复权因子，本地计算前复权/后复权
- 文档：https://tushare.pro/document/2?doc_id=28
- 权限：2000 积分起
- 关键字段：`ts_code, trade_date, adj_factor`
- 单次限制：文档未明确固定上限，建议按股票和日期分段抓取

### `pro_bar`
- 用途：A股动态复权行情 SDK 接口
- 文档：https://tushare.pro/document/2?doc_id=146
- 权限：2000 积分起
- 注意：
  - 只支持 A 股日线复权
  - 前复权以请求 `end_date` 为锚点
  - 不支持 http 方式，只能通过 Python SDK 调用

## 设计说明

- 原始缓存优先使用 `daily` + `adj_factor`
- 复权价格默认在本地静态计算，便于缓存和增量更新
- 保留 `fetch_pro_bar` 作为与官方动态复权口径对齐的校验入口
- token 仅从环境变量读取：`TUSHARE_TOKEN` 或 `TS_TOKEN`

## 缓存与限流建议

- 建议缓存目录：`data/cache/tushare/`
- 以 `ts_code` + 数据类型分文件缓存，适合逐股票增量刷新
- 调用节流建议：普通研究任务固定间隔 `0.3s` 到 `0.5s`
- 对远程请求增加至少 3 次重试，指数退避或线性退避均可
- 对全市场历史补数优先按股票或日期分批，不要一次拉全量

## 时间口径

- 本封装只提供 `trade_date` 对应的日线收盘数据
- 不改变策略时间假设
- 回测和信号生成仍应显式遵守：T 日收盘生成信号，T+1 日执行
