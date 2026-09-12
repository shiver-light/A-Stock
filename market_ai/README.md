# A股每日市场主线雷达

本模块用于盘后生成“市场主线雷达”研究报告。当前版本是离线 MVP：使用本地 CSV 行情和新闻数据，完成题材归一化、ThemeScore 排名和 Markdown 报告输出。

## 运行示例

离线示例：

```bash
/Users/raymond/src/aquant/venv/bin/python -m market_ai run \
  --trade-date 20260912 \
  --market-provider local_csv \
  --market-data-dir market_ai/examples/local_market \
  --news-csv market_ai/examples/news.csv \
  --output-dir output/market_radar_sample
```

Tushare 日线行情示例：

```bash
/Users/raymond/src/aquant/venv/bin/python -m market_ai run \
  --trade-date 20260912 \
  --market-provider tushare_daily \
  --universe-name zz500 \
  --news-csv market_ai/examples/news.csv \
  --output-dir output/market_radar_tushare_daily
```

Tushare 涨停专题示例：

```bash
/Users/raymond/src/aquant/venv/bin/python -m market_ai run \
  --trade-date 20260912 \
  --market-provider tushare_limit \
  --universe-name zz500 \
  --news-csv market_ai/examples/news.csv \
  --stock-theme-labels output/market_ai_theme_labels/20260911/limit_up_theme_labels.csv \
  --output-dir output/market_radar_tushare_limit
```

运行后会生成：

- `output/market_radar_sample/20260912.json`
- `output/market_radar_sample/20260912.md`

## 本地行情 CSV

`--market-data-dir` 目录下支持以下文件：

- `limit_stocks.csv`：涨停、触板、炸板股票。
- `strong_stocks.csv`：涨幅达到阈值的强势股票。
- `daily_quotes.csv`：预留的日行情文件，当前 MVP 不强依赖。

`limit_stocks.csv` 必需列：

- `trade_date`
- `stock_code`
- `stock_name`

`strong_stocks.csv` 必需列：

- `trade_date`
- `stock_code`
- `stock_name`
- `pct_chg`

常用可选列：

- `close`
- `amount`
- `turnover_rate`
- `volume_ratio`
- `limit_reason`
- `industry`
- `concepts`
- `consecutive_limit_count`
- `status`
- `source`

`concepts` 可使用 `|`、`,`、`，`、`;`、`；` 分隔。

## 本地新闻 CSV

`--news-csv` 指向单个新闻 CSV 文件。

必需列：

- `news_id`
- `source`
- `title`
- `published_at`

可选列：

- `url`
- `content`

`published_at` 建议使用带时区的 ISO 格式，例如 `2026-09-12T15:30:00+08:00`。

## 股票主题标签 CSV

`--stock-theme-labels` 可接入人工、概念成分或 LLM 校验后的个股主题标签。该标签优先级高于规则 taxonomy 匹配。

必需列：

- `stock_code`
- `primary_theme`

可选列：

- `trade_date`
- `secondary_themes`
- `related_theme`
- `related_entities`
- `source`
- `confidence`
- `event_driven`
- `effective_start`
- `effective_end`

示例：

```csv
trade_date,stock_code,stock_name,primary_theme,secondary_themes,related_theme,source,confidence
20260911,000823.SZ,超声电子,PCB服务器液冷电源,PCB/覆铜板,AI算力,manual,0.9
```

如果 `trade_date` 为空，则视为长期标签；如果设置 `effective_start/effective_end`，则只在有效期内参与当日归因。

## 当前假设

- 当前入口只做盘后复盘，不生成买卖建议。
- 题材归一化优先使用 `market_ai/configs/theme_taxonomy.yaml` 的确定性规则。
- 如果传入 `--stock-theme-labels`，则个股主题标签优先于 taxonomy 规则结果。
- ThemeScore 当前只使用当日行情证据和可选新闻事件。
- `tushare_daily` provider 使用 `daily(trade_date=...)` 和 `daily_basic(trade_date=...)`，涨停池暂按 `pct_chg >= 9.8` 近似。
- `tushare_daily` provider 暂不能识别炸板、盘中触板、首次封板时间、最后封板时间、开板次数、封单金额和涨停原因。
- `tushare_limit` provider 使用 Tushare `limit_list_d` 获取涨跌停和炸板数据；该接口数据从 2020 年开始，字段可用性取决于账号权限。
- `tushare_limit` provider 的强势股仍复用 `daily` / `daily_basic`，涨停原因字段暂未接入。
- `Persistence` 暂未接历史多日序列，当前为 0。
- `VolumeExpansion` 当前用当日题材成交额横截面强弱近似，后续应接 20 日均额。

## 下一步

建议后续接入真实行情 provider，例如 Tushare 涨停池和强势股 provider。业务编排层应继续依赖 `MarketProvider` 接口，避免直接绑定单一数据源。
