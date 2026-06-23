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

本数据是按小时排列的连续时间序列，测试集 ID 接在训练集之后。训练集最后一条为 `2012/8/7 11:00`，测试集第一条为 `2012/8/7 12:00`，测试集一直连续到 `2012/12/31 23:00`，共 3476 小时。因此本题更接近“用历史预测未来 3476 小时”，而不是独立随机样本回归。工程目前支持两种验证策略：

```text
random: 随机抽取 20% 作为验证集
time: 按时间顺序取最后一段作为验证集
```

随机验证会混入未来分布，之前得到的 MSE 约 900，提交反馈却仍在 3600+，说明它过于乐观。当前主流程默认 `--split-strategy time --valid-size test`，即使用训练集最后 3476 小时作为验证集，使验证 horizon 与测试 horizon 一致。若要复现旧的 20% 时间窗口，可使用 `--valid-size fraction`；若仅做对照，可使用 `--split-strategy random --valid-size fraction`。

对应实现位于 `src/data_utils.py` 的 `split_train_valid`。

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

### 4.5 连续时间特征实验

根据数据连续性，工程进一步实现了三类时间序列特征实验。

第一类是去年同期特征：`LastYearFeatureEncoder` 会构造去年同月同日同小时租借量、同比增长因子和增长调整后的去年同期先验。该特征默认关闭，可通过 `--last-year-features` 开启。

第二类是递归 lag/rolling 特征：`src/lag_features.py` 支持 `lag_1`、`lag_24`、`lag_168`、`rolling_mean_24`、`rolling_mean_168` 等特征。开启 `--lag-features` 后，验证集和测试集都会逐小时预测，并将预测值写回历史缓存，以模拟真实提交时没有未来 `cnt` 的情况。

第三类是季节性自回归残差模型：`src/seasonal_arx.py` 支持 `seasonal_arx_lightgbm` 等模型名，先构造去年同月同日同小时的季节性先验，再训练 log 残差模型，并在验证/测试阶段逐小时递归预测。

这些实验都按“不使用未来真实 cnt”的方式验证。当前结果显示它们未超过主方案：

```text
last-year features 4-model ensemble MSE: 3929.3378
last-year features tuned LightGBM MSE: 4768.8591
recursive lag features LightGBM MSE: 17799.2725
test-length seasonal_arx_lightgbm MSE: 18275.5949
```

去年同期特征在当前数据上并不是足够强的锚点，纯季节性 baseline 在测试等长验证窗口上 MSE 约 3 万；递归 lag 特征容易在早期预测偏低后向后传播误差。因此这些方法保留为实验开关，而不是默认提交路径。当前更稳的主线是：使用测试等长时间窗口重新调参静态 GBDT，并通过 raw/log 目标融合降低误差。

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

当前已按测试等长验证窗口对 LightGBM、XGBoost 及其 log 目标变体运行调参：

```bash
python src/tune_optuna.py --model lightgbm --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model lightgbm_log --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model xgboost --trials 30 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model xgboost_log --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
```

得到的结果为：

```text
test-window tuned lightgbm valid MSE: 5819.6270
test-window tuned lightgbm best iteration: 426
test-window tuned lightgbm_log valid MSE: 6110.8987
test-window tuned lightgbm_log best iteration: 622
test-window tuned xgboost valid MSE: 5733.5982
test-window tuned xgboost best iteration: 2500
test-window tuned xgboost_log valid MSE: 7254.5814
test-window tuned xgboost_log best iteration: 390
```

主训练入口可通过 `--params-dir` 自动读取调参结果：

```bash
python main.py --models lightgbm,lightgbm_log,xgboost,xgboost_log --params-dir output/params_testwindow --output output/submission_best_testwindow_tuned.csv
```

## 6. 模型融合

当同时训练多个模型时，程序会在验证集上保存每个模型的预测结果，然后用网格搜索寻找加权平均的最佳权重，使验证集 MSE 最小。

如果某个模型明显强于其他模型，融合权重会自然偏向该模型。早期 20% 时间窗口下，四模型融合结果为：

```text
hist_gradient_boosting valid MSE: 3828.6904
lightgbm valid MSE: 3743.9244
xgboost valid MSE: 3587.8082
catboost valid MSE: 4821.9862
ensemble weights: hist_gradient_boosting=0.25, lightgbm=0.05, xgboost=0.60, catboost=0.10
ensemble valid MSE: 3523.2145
```

进一步加入调参后的 raw/log 目标模型，在同一 20% 时间窗口下得到：

```text
tuned lightgbm valid MSE: 3383.7156
tuned lightgbm_log valid MSE: 3454.6130
tuned xgboost valid MSE: 3534.1538
tuned xgboost_log valid MSE: 3640.3058
ensemble weights: lightgbm=0.40, lightgbm_log=0.40, xgboost=0.15, xgboost_log=0.05
ensemble valid MSE: 3147.5131
```

随机验证策略下重新调参并融合，曾得到：

```text
random tuned lightgbm valid MSE: 953.3727
random tuned lightgbm_log valid MSE: 1037.0678
random tuned xgboost valid MSE: 937.1624
random tuned xgboost_log valid MSE: 1040.5746
ensemble weights: lightgbm=0.30, lightgbm_log=0.20, xgboost=0.40, xgboost_log=0.10
ensemble valid MSE: 898.7861
```

这些历史结果说明融合有效，但随机验证和短窗口时间验证都偏乐观。当前推荐以测试集等长时间窗口为准，该窗口下重新调参并融合后得到：

```text
lightgbm valid MSE: 5819.6270
lightgbm_log valid MSE: 6110.8987
xgboost valid MSE: 5733.5982
xgboost_log valid MSE: 7254.5814
ensemble weights: lightgbm=0.15, lightgbm_log=0.30, xgboost=0.55, xgboost_log=0.00
ensemble valid MSE: 5512.2698
```

因此当前更推荐测试等长时间窗口下调参得到的 LightGBM/XGBoost raw/log 四模型融合结果。

类别特征进一步实验显示，`catboost_cat` 会把 `season`、`mnth`、`hr`、`weekday`、`workingday`、`weathersit` 等离散列作为类别特征输入 CatBoost。未完整调参的 `catboost_cat` 在随机验证中单模型 MSE 为 `994.3666`，加入融合后得到：

```text
ensemble weights: lightgbm=0.22, lightgbm_log=0.17, xgboost=0.32, xgboost_log=0.08, lightgbm_cat=0.00, catboost_cat=0.22
random ensemble valid MSE: 890.7194
```

同一组模型在时间切分验证上为：

```text
time ensemble valid MSE: 3478.9970
catboost_cat weight: 0.00
```

这说明类别特征对随机验证有小幅收益，但随机验证和未来时间段测试之间仍存在明显分布差。由于测试集 MSE 反馈仍在 3600 左右，当前主方案切回连续最后一段验证，`catboost_cat` 暂不进入主提交融合。

另有一次历史画像特征实验：构造同月同小时同工作日等 target profile 特征后，四模型融合 MSE 退化到 `3794.4738`，说明该特征在当前时间切分下引入了分布偏差。代码中保留 `--profile-features` 作为实验开关，但默认关闭。

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
python main.py --models lightgbm,lightgbm_log,xgboost,xgboost_log --params-dir output/params_testwindow --output output/submission_best_testwindow_tuned.csv
```

如果平台接受浮点 `cnt`，可以追加 `--no-round` 生成浮点预测提交；若平台示例或规则要求整数，则保持默认四舍五入。

该方案具备以下优点：

- 已用真实数据跑通端到端流程；
- 验证方式符合时间预测场景；
- 特征覆盖日期周期、工作日、高峰期、天气和环境交互；
- 多个梯度提升树模型对非线性表格数据适配较好；
- LightGBM、XGBoost、CatBoost 已使用 early stopping 控制迭代轮数；
- LightGBM、XGBoost 及其 log 目标变体已通过 Optuna 调参；
- raw 目标模型和 log 目标模型误差形态不同，融合后验证 MSE 显著低于任一单模型；
- `catboost_cat` 显式处理类别特征，在随机验证融合中获得正权重，但在时间切分验证中权重为 0，因此暂不进入主提交；
- 输出文件满足题目要求的 `ID,cnt` 格式。

后续若继续提分，可优先尝试：

- 对递归 lag 模型单独设计更稳的基模型，避免 `lag_1` 误差扩散；
- 使用 rolling backtest 重新评估 last-year、profile 和 lag 特征；
- 继续加大 Optuna trial 数；
- 尝试 CatBoost 的类别特征模式；
- 使用 TimeSeriesSplit 检查模型在多个时间段上的稳定性；
- 做残差分析，针对高峰小时和低峰小时分别补特征。
