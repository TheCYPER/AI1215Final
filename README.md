# CreditSense

AI1215 Kaggle 竞赛：预测贷款申请人的风险等级（分类）和利率（回归）。

## 启动

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

把 `credit_train.csv` 和 `credit_test.csv` 放到 `data/` 目录下。

## 第一步：跑 EDA

```bash
python main.py --mode eda
```

会在 `outputs/eda/` 下生成 59 张图（缺失值、目标分布、每列直方图/箱线图/条形图、相关性热力图）。先看一遍数据再动手。

## 使用

```bash
# EDA（先跑这个）
python main.py --mode eda

# 训练
python main.py --mode train --task classification
python main.py --mode train --task regression

# 交叉验证
python main.py --mode cv --task classification
python main.py --mode cv --task regression

# 超参数调优（Optuna）
python main.py --mode tune --task classification
python main.py --mode tune --task regression

# 生成 Kaggle 提交文件（需要先训练好两个模型）
python main.py --mode submit
```

输出在 `outputs/` 下：EDA 图在 `eda/`，模型在 `models/`，指标在 `metrics/`，提交文件在 `predictions/`。

## 项目结构

```
configs/config.py          ← 所有超参数和配置，改这一个文件就行
data_exploration/          ← EDA，生成可视化图表
data_cleaning/             ← 数据清洗
feature_engineering/       ← 特征工程（编码器、领域特征、预处理管线）
modeling/                  ← 模型定义（BaseModel + 具体实现）
training/                  ← 训练器（单次划分 + 交叉验证）
hyperparameter_tuning/     ← Optuna 调参
submission/                ← 生成 Kaggle 提交 CSV
```

## 怎么加新东西

**换模型**：改 `configs/config.py` 里的 `clf_model_type` / `reg_model_type`。

**加新模型**：
1. 新建 `modeling/my_model.py`，继承 `BaseModel`，实现 `build_model`、`fit`、`predict`
2. 在 `modeling/__init__.py` 的 `MODEL_REGISTRY` 里注册
3. 在 `configs/config.py` 的 `ModelConfig` 里加参数

**加新特征**：改 `feature_engineering/credit_features.py`，在 `transform()` 里加新特征就行。

**加新编码器**：新建 sklearn 兼容的 Transformer，加到 `feature_engineering/preprocessor.py` 的 `build_preprocessor()` 里。
