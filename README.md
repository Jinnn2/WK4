# Bike Sharing Prediction

本项目用于完成“共享单车租借量预测”作业：根据日期、小时、天气、温度、湿度、风速等信息，预测测试集中每条记录对应小时的共享单车租借总量 `cnt`。

评价指标为均方误差 MSE，最终提交文件为 `submission.csv`，格式必须为：

```text
ID,cnt
```

## 数据说明

当前工程使用的数据文件位于：

```text
data/train.csv
data/test.csv
```

原始压缩包保留为 `data-bike.zip`，解压后的原始目录为 `data-bike/`。

训练集包含目标列 `cnt`，测试集不包含 `cnt`。真实字段如下：

```text
ID,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,cnt
```

测试集字段与训练集一致，但不含 `cnt`。

## 快速运行

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行轻量级基线模型：

```bash
python main.py --models hour_profile
```

运行当前已验证的单模型强基线：

```bash
python main.py --models hist_gradient_boosting --output output/submission.csv
```

运行当前推荐的四模型融合：

```bash
python main.py --models hist_gradient_boosting,lightgbm,xgboost,catboost --output output/submission_4model.csv
```

运行当前已调参的推荐融合：

```bash
python main.py --models lightgbm,lightgbm_log,xgboost,xgboost_log --params-dir output/params_testwindow --output output/submission_best_testwindow_tuned.csv
```

如需重新调参，可分别运行：

```bash
python src/tune_optuna.py --model lightgbm --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model lightgbm_log --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model xgboost --trials 30 --split-strategy time --valid-size test --output-dir output/params_testwindow
python src/tune_optuna.py --model xgboost_log --trials 40 --split-strategy time --valid-size test --output-dir output/params_testwindow
```

也可以单独对比 LightGBM、XGBoost、CatBoost：

```bash
python main.py --models lightgbm,xgboost,catboost
```

## 项目结构

```text
data/                  输入数据
data-bike/             原始解压数据
docs/                  文档与方案说明
notebooks/             预留 EDA 笔记本目录
output/                生成的提交文件
src/config.py          路径、列名和随机种子配置
src/data_utils.py      数据读取、字段检查、随机/时间验证切分
src/features.py        日期、周期、高峰期、天气交互等特征工程
src/lag_features.py    递归 lag/rolling 特征实验
src/seasonal_arx.py    季节性自回归残差模型实验
src/models.py          基线模型和树模型构建
src/ensemble.py        验证集加权融合搜索
src/make_submission.py 提交文件生成
src/tune_optuna.py     Optuna 调参入口
main.py                端到端训练与预测入口
```

## 解法概要

本项目将共享单车小时级需求预测建模为连续未来时间段预测问题。训练集到 `2012/8/7 11:00`，测试集从 `2012/8/7 12:00` 连续到 `2012/12/31 23:00`，共 3476 小时。为了让本地验证更接近提交场景，默认验证集也使用训练集最后 3476 小时，而不是随机抽样或固定 20% 比例。

特征工程主要包括：

- 从 `dteday` 提取年、日期、年内天数、周数、月初/月末标记；
- 对 `hr`、`mnth`、`weekday`、`dayofyear` 做周期编码；
- 构造周末、早晚高峰、通勤高峰、夜间、工作时间等业务特征；
- 构造温度、湿度、风速、天气、季节之间的交互特征。

模型方面，工程保留了轻量可解释基线 `hour_profile`，并提供 `hist_gradient_boosting`、`random_forest`、`lightgbm`、`xgboost`、`catboost` 等模型接口。LightGBM、XGBoost、CatBoost 已接入基于验证集的 early stopping，最终全量训练会复用验证阶段得到的最佳迭代轮数。

早期 20% 时间窗口验证结果如下，仅作为历史对照：

```text
hour_profile valid MSE: 15581.5756
hist_gradient_boosting valid MSE: 3828.6904
lightgbm valid MSE: 3743.9244
xgboost valid MSE: 3587.8082
catboost valid MSE: 4821.9862
4-model ensemble valid MSE: 3523.2145
tuned lightgbm valid MSE: 3383.7156
tuned lightgbm_log valid MSE: 3454.6130
tuned xgboost valid MSE: 3534.1538
tuned xgboost_log valid MSE: 3640.3058
raw/log 4-model ensemble valid MSE: 3147.5131
random-split raw/log 4-model ensemble valid MSE: 898.7861
random-split raw/log + catboost_cat ensemble valid MSE: 890.7194
```

改用测试集等长的最后 3476 小时验证后，旧参数会暴露出更明显的未来外推误差。针对该验证窗口重新调参后得到：

```text
lightgbm valid MSE: 5819.6270
lightgbm_log valid MSE: 6110.8987
xgboost valid MSE: 5733.5982
xgboost_log valid MSE: 7254.5814
ensemble weights: lightgbm=0.15, lightgbm_log=0.30, xgboost=0.55, xgboost_log=0.00
ensemble valid MSE: 5512.2698
```

因此当前推荐提交路径为：

```bash
python main.py --models lightgbm,lightgbm_log,xgboost,xgboost_log --params-dir output/params_testwindow --output output/submission_best_testwindow_tuned.csv
```

默认验证策略为连续最后一段时间切分，且 `--valid-size test` 会让验证窗口长度等于测试集长度。若要复现旧 20% 时间窗口验证，可追加 `--valid-size fraction`。若要做随机验证对照，可追加 `--split-strategy random --valid-size fraction`。如果评测平台接受浮点预测，可以追加 `--no-round` 生成不四舍五入的提交文件。

## 实验开关

仓库中保留了若干连续时间序列特征实验，但默认关闭，因为当前时间顺序验证下没有超过主方案：

```bash
python main.py --models lightgbm --params-dir output/params --last-year-features
python main.py --models lightgbm --params-dir output/params --lag-features
python main.py --models lightgbm --params-dir output/params --profile-features
python main.py --models seasonal_arx_lightgbm --params-dir output/params --output output/submission_seasonal_arx_lgbm.csv
```

已验证现象：

```text
target profile features 4-model ensemble MSE: 3794.4738
last-year features 4-model ensemble MSE: 3929.3378
last-year features tuned LightGBM MSE: 4768.8591
recursive lag features LightGBM MSE: 17799.2725
test-length seasonal_arx_lightgbm MSE: 18275.5949
```

其中递归 lag 会在验证阶段逐小时预测并回写历史，逻辑更接近真实测试；当前退化主要来自误差递归传播。`seasonal_arx_*` 进一步实现了“去年同期季节性基线 + log 残差 + 递归 lag”的论文式路线，但在本数据上去年同期基线偏弱，暂不作为推荐提交。

更完整的设计说明见 [docs/solution_design.md](docs/solution_design.md)。

## 输出文件

运行完成后会生成：

```text
output/submission_best_testwindow_tuned.csv
```

示例格式：

```text
ID,cnt
13904,269
13905,267
13906,259
```

预测值会先截断为非负数，默认再四舍五入为整数，以符合共享单车租借量的计数语义。
