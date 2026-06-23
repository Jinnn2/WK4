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
python main.py --models hist_gradient_boosting,lightgbm,xgboost,catboost --params-dir output/params --output output/submission_tuned_xgb_4model.csv
```

如需重新调参，可先运行：

```bash
python src/tune_optuna.py --model xgboost --trials 30 --output-dir output/params
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
src/data_utils.py      数据读取、字段检查、时间顺序切分
src/features.py        日期、周期、高峰期、天气交互等特征工程
src/models.py          基线模型和树模型构建
src/ensemble.py        验证集加权融合搜索
src/make_submission.py 提交文件生成
src/tune_optuna.py     Optuna 调参入口
main.py                端到端训练与预测入口
```

## 解法概要

本项目将共享单车小时级需求预测建模为表格回归问题。由于测试集时间位于训练集之后，验证集采用时间顺序切分：前 80% 数据训练，后 20% 数据验证，以模拟“用历史预测未来”的真实提交场景。

特征工程主要包括：

- 从 `dteday` 提取年、日期、年内天数、周数、月初/月末标记；
- 对 `hr`、`mnth`、`weekday`、`dayofyear` 做周期编码；
- 构造周末、早晚高峰、通勤高峰、夜间、工作时间等业务特征；
- 构造温度、湿度、风速、天气、季节之间的交互特征。

模型方面，工程保留了轻量可解释基线 `hour_profile`，并提供 `hist_gradient_boosting`、`random_forest`、`lightgbm`、`xgboost`、`catboost` 等模型接口。LightGBM、XGBoost、CatBoost 已接入基于验证集的 early stopping，最终全量训练会复用验证阶段得到的最佳迭代轮数。

当前本地已验证结果：

```text
hour_profile valid MSE: 15581.5756
hist_gradient_boosting valid MSE: 3828.6904
lightgbm valid MSE: 3743.9244
xgboost valid MSE: 3587.8082
catboost valid MSE: 4821.9862
4-model ensemble valid MSE: 3523.2145
tuned xgboost valid MSE: 3534.1538
tuned-xgb 4-model ensemble valid MSE: 3481.3420
```

因此当前推荐提交路径为：

```bash
python main.py --models hist_gradient_boosting,lightgbm,xgboost,catboost --params-dir output/params --output output/submission_tuned_xgb_4model.csv
```

更完整的设计说明见 [docs/solution_design.md](docs/solution_design.md)。

## 输出文件

运行完成后会生成：

```text
output/submission_tuned_xgb_4model.csv
```

示例格式：

```text
ID,cnt
13904,272
13905,269
13906,256
```

预测值会先截断为非负数，默认再四舍五入为整数，以符合共享单车租借量的计数语义。
