# 因子验收标准

本文档用于约束当前仓库在月度调仓框架下，如何判断一个因子或因子组合是否值得继续研究、进入荐股，或进入更接近实盘的观察阶段。

默认前提不变：

- `T` 日收盘生成信号
- `T+1` 日开盘成交
- 月度调仓
- 不静默修改：
  - `benchmark`
  - `top_n`
  - `universe`
  - `fee/slippage`
  - `execution price`

## 1. 样本覆盖要求

一个候选因子或因子组合，不能只在少数调仓点上验证。

最低要求：

- 覆盖至少 `24` 个月度调仓点
- 更推荐覆盖 `30+` 个月度调仓点
- 必须至少拆成三个时间段：
  - 探索区间
  - 验证区间
  - 保留测试区间

推荐切分：

- `20230101 ~ 20240630`
- `20240701 ~ 20250630`
- `20250701 ~ 20260331`

## 2. 单因子诊断要求

先验证因子本身，再看组合回测。

至少检查：

- `ic_mean`
- `rank_ic_mean`
- `icir`
- `coverage`
- `quantile_return_summary`

判断原则：

- `Rank IC` 方向要长期基本一致
- 分组收益方向应与因子定义大体一致
- 覆盖率不能过低
- 不能只在单一 horizon 上偶然成立

若单因子方向与组合收益明显冲突，应优先审计：

- 因子方向定义
- 极值样本驱动
- 缺失值和样本池变化

## 3. 月度表现要求

月度调仓策略必须看月度稳定性，不能只看累计收益。

至少检查：

- `positive_month_ratio`
- `positive_excess_month_ratio`

建议底线：

- `positive_month_ratio >= 0.50`
- `positive_excess_month_ratio >= 0.50`

若大多数月份不赚钱，只靠少数月份贡献收益，不视为稳健因子。

## 4. 滚动稳健性要求

必须看滚动窗口，而不是只看整段结果。

至少检查：

- `latest_rolling_5m_excess_return`
- `mean_rolling_5m_excess_return`
- `worst_rolling_5m_excess_return`
- `latest_rolling_5m_sharpe`
- `mean_rolling_5m_sharpe`
- `worst_rolling_5m_sharpe`

判断原则：

- `mean` 不应过弱
- `worst` 不应极端恶化
- `latest` 不应明显塌陷

若最差 5 个月表现极差，说明跨阶段稳健性不足。

## 5. 跨区间一致性要求

同一因子或参数不能只在探索区间有效。

最低要求：

- `explore` 区间为正
- `validate` 区间为正
- `test` 区间不明显失效

更严格的建议：

- `validate` 的 `sharpe` 不应较 `explore` 大幅恶化
- `test` 的 `excess_cumulative_return` 不应完全翻负

若只在单一区间表现突出，更可能是样本内拟合。

## 6. 跨股票池一致性要求

不是要求所有 universe 都赚钱，而是要求风格解释一致。

建议比较：

- `hs300`
- `zz500`
- `zz1000`

判断原则：

- 若在多个 universe 上方向一致，更可信
- 若只在单一 universe 成立，需明确说明其适用风格
- 若不同 universe 上方向相反，应优先怀疑因子定义或交易约束差异

## 7. 交易约束后存活要求

因子必须在更真实的交易约束下继续成立，才有资格进入下一轮。

至少要验证：

- `slippage_bps`
- `min_amount`
- 可用时验证：
  - `block_suspended`
  - `block_limit_up_buy`
  - `block_limit_down_sell`

判断原则：

- 加约束后不能直接从正超额变成完全失效
- 若约束后结果大幅恶化，应优先审计：
  - 因子是否依赖低流动性样本
  - 换手是否过高

## 8. 样本池稳定性要求

必须确认收益不是靠样本池异常收缩得到的。

至少检查：

- `universe_count_by_date`
- `factor_coverage`
- `complete_case_count_by_date`

判断原则：

- 原始 universe 数应大体稳定
- 单因子覆盖率不能长期偏低
- `complete-case` 样本不能缩到过小

若收益提升主要来自样本池急剧收缩，不视为有效 alpha。

## 9. 换手约束要求

高换手很容易让回测乐观、实盘恶化。

至少检查：

- `mean_daily_turnover`
- `mean_rebalance_turnover`
- `median_rebalance_turnover`
- `max_rebalance_turnover`

判断原则：

- 收益提升若有限，但换手明显失控，不值得保留
- `zz500 / zz1000` 对换手更敏感，要求更严格

## 10. 因子验收分级

### A级：进入下一轮实盘前观察

同时满足：

- 单因子方向基本一致
- `validate` 为正
- `test` 不明显失效
- `positive_excess_month_ratio >= 0.50`
- 交易约束后仍成立
- 换手未明显失控

### B级：继续研究

满足大部分要求，但存在一项明显短板，例如：

- 单因子方向不够干净
- `test` 区间偏弱
- 约束后收益明显下降但未完全失效

### C级：淘汰

任一情况可直接淘汰：

- 单因子方向和回测方向长期冲突
- `validate / test` 长期转负
- 加约束后直接失效
- 样本池过度收缩
- 换手严重失控

## 11. 当前仓库的实际使用建议

在当前仓库中，建议按以下顺序验收：

1. 先看 `factor_report.txt`
2. 再看 `metrics.json`
3. 再看 `summary.csv`
4. 最后才决定是否进入：
   - `recommend-consensus`
   - `watch-only`
   - `paper trading`

对当前工程，推荐的优先级是：

- `hs300`：主交易研究线
- `zz500`：增强研究线
- `zz1000`：观察研究线

## 12. 一句话原则

月度调仓因子的验收，不是看某段总收益，而是看：

- 多个月度调仓点
- 多个时间区间
- 多个稳健性指标
- 加交易约束后是否还成立
