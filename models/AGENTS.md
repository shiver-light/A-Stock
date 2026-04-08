# 模型训练规则

本目录用于机器学习模型和预测模型相关代码。

适用于：

- 因子预测模型
- 收益率预测模型
- 风险模型
- 排序模型
- 打分模型

---

# 1. 时间序列原则

必须使用时间序列划分数据。

禁止：

随机 shuffle 切分训练集和测试集。

示例：

正确：

train:
2015-2020

valid:
2021

test:
2022

错误：

sklearn random split

---

# 2. 防止未来数据泄露

特征构造必须仅使用历史数据。

禁止：

使用未来收益率构造特征
使用未来财务数据
使用未来标签信息参与特征处理

例如：

rolling 特征必须满足：

window_end <= 当前日期

---

# 3. 特征处理规则

标准化 / scaling 必须：

只在训练集上拟合

再应用到：

validation
test

示例：

正确：

fit scaler on train
transform valid/test

错误：

在全数据上 fit scaler

---

# 4. 标签构造规则

必须明确：

预测目标时间范围

例如：

未来5日收益率：

forward_return_5d

标签计算必须：

使用 shift(-n)

并确保：

不与特征时间重叠

---

# 5. 模型评估规则

评估指标需与预测目标一致：

回归：

IC
RankIC
MSE
R2

分类：

accuracy
AUC
F1

排序：

IC
Top quantile return

---

# 6. 可复现性

需要固定：

random seed

示例：

numpy
sklearn
lightgbm

需确保多次运行结果一致。

---

# 7. 特征记录

模型必须明确记录：

feature list

避免：

隐式使用 dataframe 全列。

---

# 8. 模型复杂度控制

优先使用：

线性模型
岭回归
简单树模型

再考虑：

boosting
深度学习

避免过早引入复杂模型。

---

# 9. 训练日志

应记录：

训练区间
特征数量
样本数量
评估指标

避免只输出最终收益结果。

---

# 10. agent行为约束

agent 不应：

shuffle 时间序列数据
在全样本上拟合 scaler
混用未来标签数据
默认使用复杂模型替代简单模型

agent 应：

检查时间边界
说明预测目标区间
明确特征列表
保持训练流程可复现
