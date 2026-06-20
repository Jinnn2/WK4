下面给你一份**可直接落地的实现规划**。我按“作业可完成、效果尽量好、报告可解释”的标准设计，核心路线是：

> **时间特征工程 + LightGBM / XGBoost / CatBoost + Optuna 调参 + 时间序列验证 + 模型融合 + submission.csv 输出**

---

# 共享单车租借量预测实现规划

## 一、任务目标

给定训练集 `train.csv` 和测试集 `test.csv`，训练集中包含特征列和目标列 `cnt`，测试集中只有特征列，需要预测每条测试记录对应小时的共享单车租借量。

最终提交文件格式为：

```csv
ID,cnt
13904,300
13905,300
13906,300
...
```

评价指标为均方误差：

[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
]

所以最终模型优化目标应当优先贴合 **MSE / RMSE**，而不是分类准确率或 RMSLE。

---

## 二、总体技术路线

我建议按 6 个版本迭代实现。

| 版本 | 目标          | 方法                            | 作用           |
| -- | ----------- | ----------------------------- | ------------ |
| V0 | 跑通流程        | 均值预测 / 中位数预测                  | 确认读取、训练、输出无误 |
| V1 | 建立 baseline | RandomForest / XGBoost 默认参数   | 快速得到可用结果     |
| V2 | 加入核心特征工程    | 时间、天气、工作日、高峰期特征               | 明显提升效果       |
| V3 | 三模型训练       | LightGBM + XGBoost + CatBoost | 建立强模型组       |
| V4 | 自动调参        | Optuna 分别调参                   | 提升单模型性能      |
| V5 | 模型融合        | 加权平均 / 验证集寻优权重                | 获得最终提交结果     |

最终推荐提交 **V5 融合模型**。

---

# 三、项目文件结构规划

建议你把代码组织成下面这种结构，方便调试和写报告。

```text
bike_sharing_prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission_sample.csv
│
├── output/
│   ├── submission_lgb.csv
│   ├── submission_xgb.csv
│   ├── submission_cat.csv
│   └── submission_ensemble.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── features.py
│   ├── train_lgb.py
│   ├── train_xgb.py
│   ├── train_cat.py
│   ├── tune_optuna.py
│   ├── ensemble.py
│   └── make_submission.py
│
└── main.py
```

如果你想简单一点，也可以先只写一个 `main.py`，但为了作业报告和后续调参，建议至少拆成：

```text
features.py
train.py
ensemble.py
main.py
```

---

# 四、数据处理规划

## 1. 读取数据

读取：

```python
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
```

需要检查：

```python
train.shape
test.shape
train.head()
test.head()
train.info()
train.isnull().sum()
test.isnull().sum()
```

重点确认：

1. 训练集是否有 `cnt`；
2. 测试集是否没有 `cnt`；
3. 是否有 `ID`；
4. 是否有日期时间列，例如 `datetime`、`date`、`time`；
5. 是否有泄露列，例如 `casual`、`registered`。

---

## 2. 泄露变量检查

如果数据里有下面两列：

```text
casual
registered
```

一定不能作为特征，因为：

[
cnt = casual + registered
]

这两个变量直接包含答案。如果训练时用了，会导致验证集效果虚高，但测试集无法真实泛化。

处理方式：

```python
drop_cols = ["cnt", "casual", "registered"]
```

如果没有这两列，就不用管。

---

# 五、验证集划分规划

这个题非常关键：**不要随机划分验证集**。

从截图看，训练集有 13903 条，测试集 ID 从 13904 开始，说明测试集大概率是训练集之后的连续时间段。因此应该使用时间序列验证。

## 推荐方案一：最后 20% 作为验证集

```text
训练部分：前 80%
验证部分：后 20%
```

例如：

```python
split_idx = int(len(train) * 0.8)

train_data = train.iloc[:split_idx]
valid_data = train.iloc[split_idx:]
```

这样更接近真实预测场景：用过去预测未来。

---

## 推荐方案二：TimeSeriesSplit

如果想更稳，可以使用滚动验证：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

每一折都是：

```text
前面一段训练 → 后面一段验证
```

优点是验证更稳，缺点是代码复杂一些。

对于这次作业，我建议：

> 初版用最后 20% 验证；最终版用 TimeSeriesSplit 验证模型稳定性。

---

# 六、特征工程规划

这是本题最重要的部分。

## 1. 基础特征

保留原始特征中有意义的列，例如：

```text
season
holiday
workingday
weather
temp
atemp
hum
windspeed
```

具体列名要以你的数据为准。

---

## 2. 时间特征

如果有日期时间列，例如：

```text
datetime
```

拆分为：

```python
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
```

如果没有 `datetime`，但有日期、小时等列，就根据已有列构造。

---

## 3. 周期编码特征

小时、月份、星期都是周期变量。

比如 23 点和 0 点实际上很接近，但普通数值表示里差距很大。因此加入正余弦编码：

[
hour_sin=\sin\left(\frac{2\pi hour}{24}\right)
]

[
hour_cos=\cos\left(\frac{2\pi hour}{24}\right)
]

对应代码：

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
```

---

## 4. 工作日与高峰期特征

共享单车最明显的规律是：

* 工作日：早高峰、晚高峰明显；
* 周末：中午到下午需求更高；
* 节假日：规律接近周末。

构造：

```python
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
df["is_evening_rush"] = df["hour"].isin([17, 18, 19]).astype(int)
df["is_rush_hour"] = ((df["is_morning_rush"] == 1) | 
                      (df["is_evening_rush"] == 1)).astype(int)

df["commute_rush"] = df["is_rush_hour"] * df["workingday"]
```

其中 `commute_rush` 非常重要，因为它表示：

[
通勤高峰 = 高峰小时 \times 工作日
]

这比单独的 `hour` 或 `workingday` 更能解释需求。

---

## 5. 天气交互特征

天气对骑行影响不是简单线性关系。可以加入：

```python
df["temp_hum"] = df["temp"] * df["hum"]
df["temp_windspeed"] = df["temp"] * df["windspeed"]
df["hum_windspeed"] = df["hum"] * df["windspeed"]
```

如果有 `weather`：

```python
df["bad_weather"] = (df["weather"] >= 3).astype(int)
df["weather_hour"] = df["weather"] * df["hour"]
```

如果有 `season`：

```python
df["season_hour"] = df["season"] * df["hour"]
df["season_temp"] = df["season"] * df["temp"]
```

---

## 6. 类别特征规划

对于 LightGBM 和 XGBoost，可以直接把类别特征当数值，也可以 one-hot。

建议初版先直接使用整数编码：

```text
season, holiday, workingday, weather, year, month, hour, weekday
```

对于 CatBoost，建议显式指定类别特征：

```python
cat_features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "year",
    "month",
    "hour",
    "weekday",
    "is_weekend",
    "is_rush_hour"
]
```

---

# 七、目标变量处理规划

因为评价指标是 MSE，主模型应该直接预测原始 `cnt`：

```python
y = train["cnt"]
```

但是共享单车租借量通常右偏，也可以额外训练一个 log 目标模型：

[
y_{\log} = \log(1+cnt)
]

代码：

```python
y_log = np.log1p(train["cnt"])
```

预测后反变换：

```python
pred = np.expm1(pred_log)
```

建议最终保留两类模型：

| 模型         | 目标              |
| ---------- | --------------- |
| LightGBM_A | 直接预测 `cnt`      |
| LightGBM_B | 预测 `log1p(cnt)` |
| XGBoost_A  | 直接预测 `cnt`      |
| CatBoost_A | 直接预测 `cnt`      |

然后在验证集比较：

```text
谁的 MSE 低，就给谁更高融合权重。
```

---

# 八、模型训练规划

## 1. LightGBM

LightGBM 作为主模型，优点是速度快、调参方便。

初始参数：

```python
lgb_params = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42
}
```

重点调：

```text
num_leaves
max_depth
min_data_in_leaf
learning_rate
feature_fraction
bagging_fraction
lambda_l1
lambda_l2
```

---

## 2. XGBoost

XGBoost 作为稳定 baseline。

初始参数：

```python
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42
}
```

重点调：

```text
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

---

## 3. CatBoost

CatBoost 适合处理类别变量。

初始参数：

```python
cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "bagging_temperature": 1,
    "verbose": 200,
    "random_seed": 42
}
```

重点调：

```text
depth
learning_rate
l2_leaf_reg
random_strength
bagging_temperature
```

---

# 九、Optuna 自动调参规划

## 1. 调参顺序

不要一开始就三个模型一起调。建议顺序：

```text
第一步：调 LightGBM
第二步：调 CatBoost
第三步：调 XGBoost
第四步：调融合权重
```

因为 LightGBM 速度最快，适合先找到整体参数范围。

---

## 2. 调参目标

每次 trial 返回验证集 MSE：

```python
mse = mean_squared_error(y_valid, pred_valid)
return mse
```

Optuna 目标：

```python
study = optuna.create_study(direction="minimize")
```

---

## 3. 每个模型调参次数

根据作业时间安排：

| 时间预算 | 每个模型 trial 数 |
| ---- | -----------: |
| 快速版  |           30 |
| 正常版  |           50 |
| 稳健版  |          100 |
| 极致版  |         200+ |

我建议你用：

```text
LightGBM：100 trials
CatBoost：50 trials
XGBoost：50 trials
```

这样效率和效果比较平衡。

---

# 十、融合规划

单模型可能会对某些时间段预测偏差大，所以最终使用融合。

## 1. 简单平均

最简单：

[
\hat{y}_{ens}
=============

\frac{1}{3}
(
\hat{y}*{lgb}
+
\hat{y}*{xgb}
+
\hat{y}_{cat}
)
]

代码：

```python
pred_ens = (pred_lgb + pred_xgb + pred_cat) / 3
```

---

## 2. 加权平均

更推荐：

[
\hat{y}_{ens}
=============

w_1\hat{y}*{lgb}
+
w_2\hat{y}*{xgb}
+
w_3\hat{y}_{cat}
]

其中：

[
w_1+w_2+w_3=1
]

初始权重可以设为：

```python
pred_ens = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat
```

如果验证集显示 LightGBM 最好，可以改为：

```python
pred_ens = 0.5 * pred_lgb + 0.25 * pred_xgb + 0.25 * pred_cat
```

---

## 3. 网格搜索最优融合权重

可以在验证集上搜索：

```python
best_mse = float("inf")
best_weights = None

for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        pred = w1 * pred_lgb_valid + w2 * pred_xgb_valid + w3 * pred_cat_valid
        mse = mean_squared_error(y_valid, pred)
        if mse < best_mse:
            best_mse = mse
            best_weights = (w1, w2, w3)
```

这一步非常值得做，因为简单、有效、容易写进报告。

---

# 十一、后处理规划

预测的 `cnt` 不能为负，所以需要：

```python
pred = np.maximum(pred, 0)
```

因为租借量是整数，所以提交前可以四舍五入：

```python
pred = np.round(pred).astype(int)
```

不过注意：如果平台用 MSE 直接比较，四舍五入通常影响不大；如果模型预测本来很准，保留小数也可以。但截图里的 submission 示例是整数，所以建议提交整数。

---

# 十二、输出文件规划

最终输出：

```python
submission = pd.DataFrame({
    "ID": test["ID"],
    "cnt": pred_final
})

submission.to_csv("output/submission_ensemble.csv", index=False)
```

需要确认列名严格为：

```text
ID,cnt
```

不能写成：

```text
id,count
ID,count
datetime,cnt
```

否则可能无法评分。

---

# 十三、实验记录规划

为了写报告和复盘，建议维护一张实验记录表。

| 实验编号   | 特征版本    | 模型           | 验证方式    | 验证 MSE | 备注             |
| ------ | ------- | ------------ | ------- | -----: | -------------- |
| Exp001 | 原始特征    | RandomForest | last20% |    xxx | baseline       |
| Exp002 | 时间特征    | LightGBM     | last20% |    xxx | 加 hour 后提升     |
| Exp003 | 时间+天气交互 | LightGBM     | last20% |    xxx | 加 commute_rush |
| Exp004 | V2 特征   | XGBoost      | last20% |    xxx | 稳定 baseline    |
| Exp005 | V2 特征   | CatBoost     | last20% |    xxx | 类别特征处理         |
| Exp006 | V2 特征   | LGB+XGB+CAT  | last20% |    xxx | 融合结果           |

这样写报告的时候就可以说：

> 随着时间周期、高峰期和天气交互特征的加入，模型验证集 MSE 持续下降，说明共享单车需求具有明显的时间周期性、通勤规律和天气敏感性。

---

# 十四、报告中的方法结构

你的大作业报告可以按这个逻辑写：

## 1. 问题定义

说明任务是小时级共享单车租借量回归预测，评价指标是 MSE。

## 2. 数据理解

分析训练集规模、测试集规模、特征类型、目标变量分布。

## 3. 探索性分析

画：

1. `cnt` 分布图；
2. 不同小时平均租借量；
3. 工作日与非工作日的小时租借量曲线；
4. 天气类型与租借量箱线图；
5. 温度、湿度、风速与租借量关系图。

## 4. 特征工程

说明构造了：

```text
时间特征
周期编码
工作日/周末特征
高峰期特征
天气交互特征
季节交互特征
```

## 5. 模型方法

介绍：

```text
LightGBM
XGBoost
CatBoost
Optuna
加权融合
```

## 6. 实验结果

展示不同模型 MSE。

## 7. 最终预测与提交

说明最终使用融合模型输出 `submission.csv`。

---

# 十五、推荐实现顺序

你实际写代码时，按照这个顺序最稳。

## 第一步：跑通 submission

目标：先生成一个合法的 `submission.csv`。

可以直接用训练集 `cnt` 均值预测：

```python
mean_cnt = train["cnt"].mean()
test["cnt"] = mean_cnt
```

然后输出。这样先确保文件格式没问题。

---

## 第二步：完成特征工程函数

写一个统一函数：

```python
def create_features(df):
    ...
    return df
```

要求：

```python
train_fe = create_features(train)
test_fe = create_features(test)
```

训练集和测试集必须经过完全相同的处理。

---

## 第三步：训练 LightGBM baseline

先不调参，直接训练 LightGBM，得到第一个强 baseline。

记录验证集 MSE。

---

## 第四步：加入 XGBoost 和 CatBoost

同一套特征，分别训练三个模型。

比较：

```text
LightGBM MSE
XGBoost MSE
CatBoost MSE
```

---

## 第五步：Optuna 调参

先调 LightGBM。

LightGBM 调完以后，保存最佳参数。

然后再调 CatBoost 和 XGBoost。

---

## 第六步：融合

用验证集预测结果找最优权重。

最后用全部训练集重新训练三个模型，再预测测试集。

---

# 十六、最终主程序流程

最终 `main.py` 的逻辑应该是：

```text
1. 读取 train.csv 和 test.csv
2. 合并 train/test 做统一特征工程
3. 拆回 train/test
4. 删除无用列和泄露列
5. 划分训练集和验证集
6. 训练 LightGBM
7. 训练 XGBoost
8. 训练 CatBoost
9. 验证集预测
10. 搜索最优融合权重
11. 使用全量训练集重新训练模型
12. 对 test 进行预测
13. 融合预测结果
14. 后处理：非负、整数化
15. 生成 submission.csv
```

---

# 十七、风险点和对应解决方案

| 风险          | 表现                   | 解决方案                           |
| ----------- | -------------------- | ------------------------------ |
| 随机划分验证集导致虚高 | 本地分数很好，提交很差          | 使用时间序列划分                       |
| 使用泄露变量      | 验证 MSE 异常低           | 删除 `casual`, `registered`      |
| 特征列不一致      | 训练时报错                | train/test 合并后统一特征工程           |
| 类别特征处理不当    | CatBoost 报错          | 指定 `cat_features` 并转为 int/str  |
| 预测出现负数      | submission 中 cnt < 0 | `np.maximum(pred, 0)`          |
| 输出格式错误      | 无法提交                 | 严格输出 `ID,cnt`                  |
| 过拟合         | 训练误差低、验证误差高          | 增大 `min_data_in_leaf`，减小树深，加正则 |
| 测试集未来分布变化   | 验证不稳定                | 使用 TimeSeriesSplit 或最后 20% 验证  |

---

# 十八、推荐最终方案摘要

最终实现可以概括成：

> 本项目采用面向结构化时间序列数据的梯度提升树集成方法。首先对日期、小时、星期、工作日、节假日、天气、温度、湿度、风速等变量进行特征工程，构造周期编码、高峰期、通勤高峰、天气交互和季节交互特征。随后分别训练 LightGBM、XGBoost 和 CatBoost 三类梯度提升树模型，并使用时间序列验证集评估模型泛化能力。进一步利用 Optuna 对关键超参数进行自动搜索，最后根据验证集 MSE 确定三模型融合权重，生成最终的共享单车租借量预测结果。

