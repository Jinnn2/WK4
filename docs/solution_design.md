# 共享单车租借量预测解法设计

## 1. 任务理解

本任务要求根据每小时的日期、时间、天气和环境变量，预测该小时共享单车租借总量 `cnt`。训练集包含目标列 `cnt`，测试集不包含目标列，需要输出每条测试样本的预测结果。

评价指标为均方误差 MSE：

```text
MSE = mean((y_true - y_pred)^2)
```

因此模型应直接优化连续数值回归效果，而不是分类准确率或排序指标。

## 2. 数据与字段

真实训练集字段为：

```text
ID,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,cnt
```

测试集字段与训练集一致，但没有 `cnt`。

字段含义可以分为四类：

- 标识字段：`ID`
- 时间字段：`dteday`、`yr`、`mnth`、`hr`、`weekday`
- 日历字段：`holiday`、`workingday`、`season`
- 天气环境字段：`weathersit`、`temp`、`atemp`、`hum`、`windspeed`

其中 `ID` 和 `dteday` 不直接作为普通数值特征输入模型，`dteday` 会先被拆解为更有意义的日期特征。目标列 `cnt` 只在训练时使用，不能作为输入特征。

## 3. 验证策略

本数据是按小时排列的时间序列式表格数据，测试集 ID 接在训练集之后。因此验证时不采用随机切分，而采用时间顺序切分：

```text
前 80% 训练
后 20% 验证
```

这样更接近真实提交场景：用较早的历史记录预测较晚的未来记录。随机切分虽然可能得到更乐观的验证结果，但会混入未来时间段的信息，不利于评估模型的真实泛化能力。

对应实现位于 `src/data_utils.py` 的 `time_order_split`。

## 4. 特征工程

共享单车需求具有明显的时间周期性、通勤规律和天气敏感性，因此特征工程是本方案的核心。

### 4.1 日期特征

从 `dteday` 中提取：

- `year_abs`
- `day`
- `dayofyear`
- `weekofyear`
- `is_month_start`
- `is_month_end`

这些特征帮助模型识别长期趋势、月内位置和季节变化。

### 4.2 周期编码

对周期变量使用 sin/cos 编码：

- 小时：`hr_sin`、`hr_cos`
- 月份：`mnth_sin`、`mnth_cos`
- 星期：`weekday_sin`、`weekday_cos`
- 年内天数：`dayofyear_sin`、`dayofyear_cos`

周期编码可以表达“23 点和 0 点很接近”“12 月和 1 月很接近”这类循环关系，避免模型把它们理解成普通线性距离。

### 4.3 工作日与高峰期特征

构造的业务特征包括：

- `is_weekend`
- `is_morning_rush`
- `is_evening_rush`
- `is_rush_hour`
- `is_night`
- `is_work_hour`
- `commute_rush`

共享单车租借量通常受上下班通勤影响明显，因此早高峰、晚高峰、工作日通勤高峰是很重要的解释变量。

### 4.4 天气与环境交互特征

构造的交互特征包括：

- `temp_hum`
- `temp_windspeed`
- `hum_windspeed`
- `feels_temp_gap`
- `bad_weather`
- `weather_hour`
- `season_hour`
- `season_temp`

这些特征用于表达非线性影响。例如同样的温度，在高湿度或大风条件下对骑行意愿的影响可能不同；同样的坏天气，出现在通勤高峰和深夜的影响也不同。

对应实现位于 `src/features.py`。

## 5. 模型设计

工程中保留多层模型，以便从可运行基线逐步提升效果。

### 5.1 轻量基线

`mean` 模型直接预测训练集均值，用于检查流程是否可运行。

`hour_profile` 模型按照月份、小时、工作日和天气等字段分组统计历史平均租借量。它不依赖额外机器学习库，可解释性强，适合作为端到端基线。

### 5.2 单模型强基线

`hist_gradient_boosting` 是当前本地已验证的强基线。它属于梯度提升树模型，适合处理表格数据中的非线性关系和特征交互。

本地时间顺序验证结果：

```text
hour_profile valid MSE: 15581.5756
hist_gradient_boosting valid MSE: 3828.6904
```

在不安装额外模型依赖或需要快速生成提交文件时，可以使用：

```bash
python main.py --models hist_gradient_boosting --output output/submission.csv
```

### 5.3 可选增强模型

工程还预留了：

- `random_forest`
- `lightgbm`
- `xgboost`
- `catboost`

其中 LightGBM、XGBoost、CatBoost 是表格回归任务中常用的梯度提升树模型。安装相关依赖后，可以用同一套特征训练并比较验证集 MSE。

当前工程已为 LightGBM、XGBoost、CatBoost 接入基于验证集的 early stopping。验证阶段会记录最佳迭代轮数，最终全量训练时会用该轮数重新训练，避免默认轮数过大导致过拟合。

当前本地验证结果：

```text
lightgbm valid MSE: 3743.9244
lightgbm best iteration: 379
xgboost valid MSE: 3587.8082
xgboost best iteration: 1275
catboost valid MSE: 4821.9862
catboost best iteration: 1223
```

### 5.4 Optuna 调参

工程已实现 `src/tune_optuna.py`，用于对 LightGBM、XGBoost、CatBoost 做自动超参数搜索。调参脚本会在时间顺序验证集上最小化 MSE，并把最优参数保存为 JSON 文件。

当前已对 XGBoost 运行 30 次 trial：

```bash
python src/tune_optuna.py --model xgboost --trials 30 --output-dir output/params
```

得到的结果为：

```text
tuned xgboost valid MSE: 3534.1538
best iteration: 569
params file: output/params/xgboost.json
```

主训练入口可通过 `--params-dir` 自动读取调参结果：

```bash
python main.py --models hist_gradient_boosting,lightgbm,xgboost,catboost --params-dir output/params --output output/submission_tuned_xgb_4model.csv
```

## 6. 模型融合

当同时训练多个模型时，程序会在验证集上保存每个模型的预测结果，然后用网格搜索寻找加权平均的最佳权重，使验证集 MSE 最小。

如果某个模型明显强于其他模型，融合权重会自然偏向该模型。当前已验证的四模型融合结果为：

```text
hist_gradient_boosting valid MSE: 3828.6904
lightgbm valid MSE: 3743.9244
xgboost valid MSE: 3587.8082
catboost valid MSE: 4821.9862
ensemble weights: hist_gradient_boosting=0.25, lightgbm=0.05, xgboost=0.60, catboost=0.10
ensemble valid MSE: 3523.2145
```

融合后的验证 MSE 低于任一单模型。进一步使用调参后的 XGBoost 参与融合，得到：

```text
hist_gradient_boosting valid MSE: 3828.6904
lightgbm valid MSE: 3743.9244
tuned xgboost valid MSE: 3534.1538
catboost valid MSE: 4821.9862
ensemble weights: hist_gradient_boosting=0.15, lightgbm=0.00, xgboost=0.70, catboost=0.15
ensemble valid MSE: 3481.3420
```

因此当前更推荐调参 XGBoost 参与的四模型融合结果。

对应实现位于 `src/ensemble.py`。

## 7. 预测后处理与提交

模型预测值经过两步后处理：

1. 将负数预测截断为 0；
2. 默认四舍五入为整数。

原因是 `cnt` 表示租借数量，不应为负数，且提交示例为整数计数。

最终输出文件为：

```text
output/submission.csv
```

文件格式为：

```text
ID,cnt
13904,272
13905,269
13906,256
```

对应实现位于 `src/make_submission.py`。

## 8. 当前推荐方案

当前推荐提交方案为：

```bash
python main.py --models hist_gradient_boosting,lightgbm,xgboost,catboost --params-dir output/params --output output/submission_tuned_xgb_4model.csv
```

该方案具备以下优点：

- 已用真实数据跑通端到端流程；
- 验证方式符合时间预测场景；
- 特征覆盖日期周期、工作日、高峰期、天气和环境交互；
- 多个梯度提升树模型对非线性表格数据适配较好；
- LightGBM、XGBoost、CatBoost 已使用 early stopping 控制迭代轮数；
- XGBoost 已通过 Optuna 完成一轮 30-trial 调参；
- 验证集加权融合 MSE 低于当前任一单模型，并进一步低于未调参四模型融合；
- 输出文件满足题目要求的 `ID,cnt` 格式。

后续若继续提分，可优先尝试：

- 使用 Optuna 对树模型调参；
- 继续分别调参 LightGBM 和 CatBoost；
- 使用 TimeSeriesSplit 检查模型在多个时间段上的稳定性；
- 比较单模型、log 目标模型和多模型融合的验证集 MSE。
