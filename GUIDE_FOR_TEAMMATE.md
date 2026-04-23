# 跑 TabularTransformer 实验的指南（给队友）

Percy 版 repo: <https://github.com/TheCYPER/AI1215Final>

你仓库里的 `TabularTransformer` 已经被搬到这个 repo，注册进了我们的
`BaseModel` 工厂。现在它跟 CatBoost / TabNet / CORN-MLP 共享同一套
CV / stacking / tuning / submission 管线。以下指南假定你用
CUDA 环境继续跑真正的实验。

---

## 0. 我这边 SOTA 的真实结构

我的 cls SOTA 不是单纯的 12-TabNet stacking，而是一个 **gated mixture**：

```
Stack(12 TabNet + 5 CORN-MLP) + LogReg meta      ← "jury"
          ↑
          └─ 低置信 (top1 proba < 0.59) 时 ──> TabNet single (tuned)  ← "judge"
```

- **Stacking base** (5-fold CV): 0.8679 ± 0.0040
  - 配置: `configs/config.py` 的 `tabnet_plus_coral_components(5)`
  - 17 个 base: 12 TabNet (8 tuned seed-jitter + 3 wild + 1 baseline)
    + 5 CORN-MLP (hidden 256/512/192/384/128 变体)
  - Meta learner: LogReg，80/20 holdout 下训练
- **Gated SOTA** (OOF + half-split 验证): 0.8718
  - Threshold t=0.59，用独立的 half-split 二次验证过
  - 覆盖实验: `experiments.md` row #47 (stack) + row #48 (gated)
  - 生成提交文件: `scripts/generate_gated_submission.py`

所以你看到我说 "5 MLP + 12 TabNet + 专门训练的 TabNet 做 gate" 就是这个结构：
5 个 CORN-MLP 就是 "5 MLP"（CORN 是 ordinal 版的 MLP）；"专门的 TabNet"
就是 TabNet-single 作为 low-confidence specialist。

**回归 SOTA**: 0.8406 ± 0.0152，纯 12-TabNet stacking
(`pure_tabnet_reg_stacking_components`)；reg 目前 gating 没意义，row #51
的两阶段 (cls OOF 注入 reg feature) 也到 ceiling 0.842。

---

## 1. 主要思路：把两边 SOTA 合起来跑 Ensemble

两边架构完全不同 —— TabNet 是 sparse-attention 的 sequential decision
process，Transformer 是 dense self-attention over feature tokens；
CORN-MLP 是 feedforward + ordinal head —— 三种归纳偏置正交。按
`experiments.md` row #40 的错题分析结论，跨架构组合是唯一还没被我耗尽
的破 SOTA 路径。

**主攻目标**（优先级从高到低）:

1. **把 Transformer 塞进 stacking base**: 17 → 17+N，看 CV 能不能过
   0.8679，以及 gated 能不能过 0.8718。
2. **用 Transformer 当 gate specialist**: 替代或补充 TabNet single 的
   "judge" 角色，看 gated 阈值扫描能不能找到更好的曲线。
3. **回归版本**: 同样把 Transformer 塞进 12-TabNet reg stack，看能不能
   过 0.8406。

具体流程见 §6。先把环境和超参搞好。

---

## 2. 拉代码 + 环境

```bash
git clone https://github.com/TheCYPER/AI1215Final.git
cd AI1215Final
git checkout main
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

数据放这里（和你原 repo 一样的三个 CSV）：

```
data/credit_train.csv        # 35k × 57
data/credit_test.csv         # 15k × 55
data/sample_submission.csv
```

验证 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. 先跑 smoke test

纯 wiring 检查，CPU 一秒过：

```bash
python scripts/smoke_tabular_transformer.py
# all smoke checks passed
```

这只验证 fit / predict / predict_proba 形状和数值正常 —— 不看分数。
让你确认一下拉下来的 repo 能跑。

---

## 4. 把你的 Transformer SOTA 超参应用到这边

你在原 repo 跑过 Optuna，应该有一份 `best_config.json`（那边
`results/tuning_cls/best_config.json` 或类似路径）。

**选项 A**（推荐，最快）: 直接改 `configs/config.py`

找到 `tabular_transformer_clf_params`，把你的 best_params 覆盖进去：

```python
tabular_transformer_clf_params: Dict[str, Any] = field(default_factory=lambda: {
    # 把你 Optuna best_config.json 的值粘进来：
    "d_model": <你的>,
    "n_heads": <你的>,
    "n_layers": <你的>,
    "d_ff": <你的>,
    "dropout": <你的>,
    "num_embedding_type": <你的>,   # "numerical" / "periodic" / "ple"
    "use_cls_token": <你的>,
    "lr": <你的>,
    "weight_decay": <你的>,
    "batch_size": <你的>,
    # 其余保持默认：
    "pooling": "cls", "pre_norm": True,
    "use_column_embedding": False,
    "use_ordinal": True,           # CORN 头；建议开，破 softmax ~1pt
    "max_epochs": 100, "patience": 15,
    "grad_clip": 1.0, "seed": 42,
})
```

回归同理，改 `tabular_transformer_reg_params`。

**选项 B**: 重新跑 Optuna（耗时；只在想要我们 preprocessor 下的重新
tune 时用）。见 §8。

---

## 5. 跑单模型 CV 生成 OOF

超参改好后跑一次 5-fold CV —— 目的有两个：
(a) 看 single Transformer 在我们的 preprocessor 下的分数；
(b) 保存 OOF，后面 stacking / gating 实验要用。

### 分类

```bash
python main.py --mode cv --task classification --model tabular_transformer
```

### 回归

```bash
python main.py --mode cv --task regression --model tabular_transformer
```

4090 上每次 5-fold 估计 **15–30 分钟**（看 `max_epochs`）。

输出：

- `outputs/metrics/{classification,regression}_cv_summary.json` ← fold mean / std
- `outputs/oof/tabular_transformer.npz` ← OOF preds（stacking / gating 用）
- `outputs/analysis/` ← cls 自动跑的错题分析

**经验判据（来自 row #46-47）**: 单模型比 TabNet single (0.8573) 差
> 1pt，塞进 stacking 通常稀释而非帮助。Single ≥ 0.855 → §8 combined
stacking 值得跑；< 0.845 → 先回 §10 tune。

---

## 6. 复现我这边 cls SOTA（推荐先跑一次）

跑完后你会有三份 OOF 文件（`outputs/oof/cls_stack_12tn_5coral.npz` /
`cls_tabnet_single.npz` / `cls_sota_gated.npz`），是 §7 error analysis
和 §8 gated 扫描的输入。不先跑这个，下游实验没 baseline 对比，
`analysis/error_correlation.py` 的 Q 矩阵也会缺一大半 base。

### 6.1 一条命令跑完（推荐）

我写过一个 5 步编排器 `scripts/two_stage_v2_with_sota_oof.py`，从 cls
stack CV → TabNet single CV → gated OOF 合成 → reg 两阶段 CV 全自动：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    nohup python -u scripts/two_stage_v2_with_sota_oof.py \
    > /tmp/two_stage_v2.log 2>&1 &

# 跟进进度
tail -f /tmp/two_stage_v2.log
```

4090 上总耗时 **~13 小时**（cls stack 5-6h + cls TabNet 1.5-2h +
gated 秒级 + reg stack ~5h）。期望输出：

| Step | Target | 说明 |
|------|--------|------|
| 1 | acc ≈ 0.8679 | `outputs/oof/cls_stack_12tn_5coral.npz` |
| 2 | acc ≈ 0.8573 | `outputs/oof/cls_tabnet_single.npz` |
| 3 | acc ≈ 0.8718 | `outputs/oof/cls_sota_gated.npz`（**SOTA**） |
| 4 | — | `data/credit_train_cls_aug_v2.csv` |
| 5 | r² ≈ 0.8422 | 回归两阶段（ceiling 验证）|

如果只要 cls SOTA 不想等 reg：编辑该脚本底部的 `pipeline` 列表，
把最后两项 (`build_v2_aug`, `reg_two_stage_v2`) 删掉再跑。cls 部分 ~7 小时搞定。

### 6.2 手动分步（只想要 cls SOTA）

**Step 1: cls stack(12 TabNet + 5 CORN-MLP) 5-fold CV**（~5-6 h）

```bash
python scripts/stack_tabnet_coral.py --n_coral 5 --tag cls_stack_12tn_5coral
```

结束后 OOF 自动存到 `outputs/oof/cls_stack_12tn_5coral.npz`，summary
在 `outputs/metrics/cls_stack_12tn_5coral_summary.json`。对标 row #47
的 0.8679 ± 0.0040。

**Step 2: cls TabNet single 5-fold CV**（~1.5-2 h）

```bash
python main.py --mode cv --task classification --model tabnet
# 默认存在 outputs/oof/tabnet.npz，改名方便后续脚本认：
mv outputs/oof/tabnet.npz outputs/oof/cls_tabnet_single.npz
```

对标 row #24 的 0.8573 ± 0.0049。

**Step 3: 合成 gated OOF**（秒级）

```bash
python -c "
import numpy as np
s = np.load('outputs/oof/cls_stack_12tn_5coral.npz')
t = np.load('outputs/oof/cls_tabnet_single.npz')
order_s = np.argsort(s['indices'])
order_t = np.argsort(t['indices'])
P_s, P_t = s['y_proba'][order_s], t['y_proba'][order_t]
y_true = s['y_true'][order_s]
assert np.array_equal(y_true, t['y_true'][order_t]), 'y_true mismatch'

threshold = 0.59
mask = P_s.max(axis=1) < threshold
P_g = np.where(mask[:, None], P_t, P_s)
pred = P_g.argmax(axis=1)
acc_s = (P_s.argmax(1) == y_true).mean()
acc_t = (P_t.argmax(1) == y_true).mean()
acc_g = (pred == y_true).mean()
print(f'stack acc  = {acc_s:.4f}  (target ~0.8679)')
print(f'tabnet acc = {acc_t:.4f}  (target ~0.8573)')
print(f'gated acc  = {acc_g:.4f}  (target ~0.8718)  <-- SOTA')
print(f'low-conf fallback: {mask.mean()*100:.1f}%  (~6-7%)')

np.savez_compressed('outputs/oof/cls_sota_gated.npz',
    y_true=y_true, y_pred=pred, y_proba=P_g,
    indices=np.arange(len(y_true)))
print('saved → outputs/oof/cls_sota_gated.npz')
"
```

### 6.3 （可选）全数据训练 + 生成提交 CSV

Repro 过了就可以直接做 Kaggle 提交：

```bash
# 前置条件: outputs/models/regression_pipeline.joblib 存在（回归 pipeline）
# 没有的话先跑 scripts/train_reg_on_full_data.py

python scripts/generate_gated_submission.py \
    --threshold 0.59 \
    --reg_pipeline outputs/models/regression_pipeline.joblib \
    --output outputs/predictions/submission_sota.csv
```

这个脚本做的事：
1. 在全量 train 上 fit `tabnet_plus_coral_components(5)` stacking
2. 在全量 train 上 fit TabNet single
3. test 集预测双路，`stack top1 < 0.59` 则用 TabNet 的 argmax
4. 拼 `Id, RiskTier, InterestRate`（回归直接从 `--reg_pipeline` 加载）
5. 写 submission CSV，也存 `outputs/models/gated_cls_artifact.joblib` 便于复用

4090 上 ~1.5-2 h（只训一次、不 CV）。

### 6.4 快速 sanity check（不想等全跑）

如果想先看 stack 和 TabNet single 单独靠不靠谱再决定要不要等全 CV：

```bash
# ~1 h：只跑 1 fold 的 stack 就能看架构通不通
python -c "
from configs.config import Config, TaskType, tabnet_plus_coral_components
from training.cross_validator import CrossValidator
cfg = Config()
cfg.training.task_type = TaskType.CLASSIFICATION
cfg.training.n_splits = 2  # 2-fold 快速 check
cfg.models.clf_model_type = 'ensemble'
cfg.models.ensemble_clf_components = tabnet_plus_coral_components(5)
cfg.models.ensemble_clf_mode = 'stacking'
cfg.models.ensemble_meta_learner_type = 'logreg'
summary = CrossValidator(cfg).run()
print(summary)
"
```

2-fold 应该 >0.86（比 5-fold 略高，因为每折 train 数据更多）。明显低说明环境有问题。

---

## 7. SOTA 的 error analysis

分两层：**per-model 错题分析**（每次 CV 自动跑）+ **跨模型多样性**
（`analysis/error_correlation.py` 手动跑）。gated SOTA 因为不是 CV
内直接产物，得手动对 `cls_sota_gated.npz` 单独跑。

### 7.1 单模型错题分析

`training/cross_validator.py` 每次 cls CV 跑完**自动**调用
`analysis/error_analyzer.run_error_analysis`，生成：

```
outputs/analysis/
├── confusion_matrix.png      # 5×5 计数 + row-normalized 热图
├── hard_examples.csv         # 100 个高置信预测错 → 最难样本
└── error_analysis.json       # per-class P/R/F1 + confusion_pairs + confidence bins
```

`error_analysis.json` 几个关键字段：

- `confusion_pairs["<class>"]` — 每个 class 最常被错分成哪个 class + count
- `per_class` — sklearn classification_report（precision / recall / f1 per class）
- `confidence_analysis.bins` — 按 max-softmax 分 10 段，每段的 n / accuracy /
  mean_confidence（做校准 & 挑 gate threshold 的依据）

注意这些会被下一次 CV 覆盖。想保留就 `cp -r outputs/analysis outputs/analysis_<tag>`。

### 7.2 对 gated SOTA 单独跑（手动）

Gated 不在 CV 流程里，要自己调：

```bash
mkdir -p outputs/analysis/cls_sota_gated
python -c "
import numpy as np
from analysis.error_analyzer import run_error_analysis

d = np.load('outputs/oof/cls_sota_gated.npz')
res = run_error_analysis(
    y_true=d['y_true'], y_pred=d['y_pred'], y_proba=d['y_proba'],
    original_indices=d['indices'],
    out_dir='outputs/analysis/cls_sota_gated',
    n_classes=5,
)
print('overall acc:', res['confidence_analysis']['overall_accuracy'])
print('mean confidence:', res['confidence_analysis']['overall_mean_confidence'])
for k, v in res['confusion_pairs'].items():
    print(f'  class {k} most confused with class {v[\"most_confused_with\"]} '
          f'({v[\"count\"]}/{v[\"total_errors\"]} errors)')
"
```

Row #40 的主要结论（供参考）：RiskTier 错题基本是 ±1 相邻类，
极少跳 2 格以上 → 所有模型都在 boundary 上犹豫，这也是 ordinal head
（CORN）有效的原因。

### 7.3 跨模型多样性分析（最有用）

这是选要不要把某个新 base 加进 stacking 的核心工具。读 `outputs/oof/`
下所有 `.npz` 配对算 Yule's Q / error-overlap：

```bash
python -m analysis.error_correlation
```

输出：

```
outputs/analysis/
├── error_correlation.md                # per-model acc + Q matrix + overlap matrix
├── error_correlation.json              # 同上 JSON
└── error_correlation_heatmap.png       # Yule's Q 热图
```

**读 Q 矩阵要点**（越低越好，对 ensemble 而言）：

- Q = +1 完美一致 → 加它完全无用
- Q = 0 独立 → 理论最佳互补
- Q = -1 完美反向 → 实际很难达到

已知的 baseline Q 值（row #40 测过）：

| pair | Q | 解读 |
|------|---|------|
| tree vs tree (XGB, LGBM, CatBoost) | 0.96–0.97 | 同架构挤爆 |
| TabNet vs 所有 tree | 0.90–0.93 | 有一点空间 |
| CORN-MLP vs TabNet | 0.938 | 意外地高 |
| logreg_poly vs TabNet | 0.784 | 最低，但 acc 太差用不了 |

**你的 Transformer 入矩阵后的判据**:

- `Q(tabular_transformer, tabnet) < 0.90` → 跨架构多样性成立，加进 stacking
  大概率有用
- 0.90 ≤ Q < 0.94 → 边界区，看 single acc 强不强决定
- Q ≥ 0.94 → Transformer 学到的决策边界和 TabNet 差不多，加进去会稀释

**前置**: 先让 OOF 目录里有 Transformer OOF（§5 跑完 single CV 就有）
+ 我 SOTA 的三份 OOF（§6 跑完就有）。目录里至少要有 2 个 `.npz` 才能跑。

### 7.4 置信度分箱（验证 gate 阈值）

Gate threshold 0.59 不是拍脑袋 —— row #48 在 [0.50, 0.99] 上扫
confidence bin 挑的平顶区间。你要自己看：

```python
from analysis.error_analyzer import confidence_analysis
import numpy as np, json

d = np.load('outputs/oof/cls_stack_12tn_5coral.npz')
print(json.dumps(
    confidence_analysis(d['y_true'], d['y_proba'], bins=20),
    indent=2,
))
```

每 bin 的 acc 在 ~0.55-0.65 一段会掉一截 —— 那就是 stack 开始不靠谱、
该交给 specialist 的阈值。`half-split 独立 tune 两半都选到 0.590`
就是行 #48 记的那个稳健性证据。

### 7.5 挑 specialist（row #45 的教训）

Row #45 扫过 6 个 specialist × 50 档 threshold，**关键发现**：
gated 收益严格按 specialist 单模型 acc 排序，与 Q 几乎无关。

所以你要用 Transformer 当 specialist（§8 §6.2）的时候，先看它
single-model acc 够不够高（≥0.85 起步），Q 的话再差一点也无所谓。

---

## 8. 核心实验：把 Transformer 加进 SOTA

### 8.1 加进 stacking base（第一优先级）

现在的 stacking base 是 `tabnet_plus_coral_components(5)` = 12 TabNet
+ 5 CORN-MLP = 17 bases。写一个 generator 加 N 个 Transformer 进去：

在 `configs/config.py` 里（跟 `tabnet_plus_coral_components` 放一起）加：

```python
def tabnet_coral_plus_transformer_components(
    n_transformer: int = 3,
) -> List[Dict[str, Any]]:
    """SOTA stacking base (12 TabNet + 5 CORN-MLP) + N TabularTransformer.

    假设：Transformer 的 dense self-attention 归纳偏置与 TabNet 的
    sparse sequential attention + CORN-MLP 的 feedforward 正交（row #40
    错题分析说明需要换架构才能真正 decorrelate），LogReg meta 在扩展的
    (17+N)*5 维 OOF proba 上应该能找到利用 Transformer 的方式。
    """
    components = tabnet_plus_coral_components(5)  # 17 bases
    for i in range(n_transformer):
        components.append({
            "type": "tabular_transformer",
            "overrides": {
                "seed": 500 + i,
                # 让 N 个 variant 走不同 numerical embedding，放大多样性：
                "num_embedding_type": ["numerical", "periodic", "ple"][i % 3],
            },
        })
    return components
```

然后写一个驱动脚本，类似 `scripts/stack_tabnet_coral.py`：

```python
# scripts/stack_tabnet_coral_transformer.py
from configs.config import Config, TaskType, tabnet_coral_plus_transformer_components
from training.cross_validator import CrossValidator

cfg = Config()
cfg.training.task_type = TaskType.CLASSIFICATION
cfg.training.n_splits = 5
cfg.models.clf_model_type = "ensemble"
cfg.models.ensemble_clf_components = tabnet_coral_plus_transformer_components(3)
cfg.models.ensemble_clf_mode = "stacking"
cfg.models.ensemble_meta_learner_type = "logreg"
cfg.models.ensemble_stack_method = "holdout"

cv = CrossValidator(cfg)
summary = cv.run()
print(summary)
```

4090 上 5-fold 预计 **2.5–3.5 小时**（17 TabNet+CORN-MLP + 3 Transformer
base × 5 fold）。结果对标 0.8679（stacking base）。

### 8.2 再挂上 gate → 对标 0.8718

跑完 §8.1 CV 后，`outputs/oof/` 下会有新 stack OOF 文件。用这个 OOF
做一轮 gated mixture 扫描（同 row #48 模板），对标 0.8718。

核心逻辑（参考 `scripts/generate_gated_submission.py` 的 gating 部分）：

```python
# 伪代码：扫 threshold
import numpy as np
stack_oof = np.load("outputs/oof/<your_new_stack_tag>.npz")
tabnet_oof = np.load("outputs/oof/tabnet.npz")          # 已有的 TabNet single OOF
# 可以再换成 transformer single OOF 试试：
# transformer_oof = np.load("outputs/oof/tabular_transformer.npz")

stack_proba = stack_oof["y_proba"]
specialist_proba = tabnet_oof["y_proba"]
y_true = stack_oof["y_true"]
top1 = stack_proba.max(axis=1)
for t in np.linspace(0.40, 0.80, 41):
    mask = top1 < t
    pred = np.where(mask, specialist_proba.argmax(1), stack_proba.argmax(1))
    acc = (pred == y_true).mean()
    print(f"t={t:.3f}  acc={acc:.4f}  low_conf_rate={mask.mean():.3f}")
```

Specialist 可选 TabNet single（当前 SOTA 里的 judge）或 Transformer
single（新候选）—— 两个都扫一下。

### 8.3 全数据训练 + Kaggle 提交

如果 CV + gated 确实破了 0.8718，改 `scripts/generate_gated_submission.py`：

- 把第 37 行 `tabnet_plus_coral_components(5)` 换成
  `tabnet_coral_plus_transformer_components(3)`（或你最终选的 N）。
- 如果 gate specialist 换成 Transformer single，把第 116-122 行的
  `"tabnet"` 改成 `"tabular_transformer"`。
- 跑:
  ```bash
  python scripts/generate_gated_submission.py \
      --threshold <你扫出来的> \
      --reg_pipeline outputs/models/regression_pipeline.joblib \
      --output outputs/predictions/submission_new.csv
  ```

### 8.4 回归版本

`pure_tabnet_reg_stacking_components` 是 12-TabNet reg stack (row #49,
r² 0.8406)。加 Transformer:

```python
def tabnet_plus_transformer_reg_components(
    n_transformer: int = 3,
) -> List[Dict[str, Any]]:
    components = pure_tabnet_reg_stacking_components()
    for i in range(n_transformer):
        components.append({
            "type": "tabular_transformer",
            "overrides": {
                "seed": 600 + i,
                "num_embedding_type": ["numerical", "periodic", "ple"][i % 3],
            },
        })
    return components
```

应用：

```python
ensemble_reg_components: List[Any] = field(
    default_factory=lambda: tabnet_plus_transformer_reg_components(n_transformer=3)
)
ensemble_reg_mode: str = "stacking"
ensemble_meta_learner_type: str = "ridge"
```

跑：

```bash
python main.py --mode cv --task regression --model ensemble
```

对标 reg SOTA 0.8406。row #51 指出 reg 的 ceiling 可能就是 0.842
(irreducible noise)，但架构多样性有机会把天花板推上去一点。

### 8.5 N 的扫描

第一轮推荐 `n_transformer=3`。如果 +3 拉过 SOTA 了，可以再试 5 / 7
看边际收益；如果 +3 微降，试 1 / 2 看稀释规律。这个是 row #46-47 对
CORN-MLP 做过的 ablation（3 → 5 CORN-MLP 只 +0.0004）。

---

## 9. 实验记录

每跑完一个 CV，在 `experiments.md` 末尾加一行。表格的 9 列：

| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 | 备注 |

**结论判准**（`ml-experiment-loop` skill 的默认定义）：

- `work`: delta > 2 × baseline std
- `待定`: |delta| < 1 × baseline std
- `not work`: delta < -2 × baseline std

也可以用 `python .claude/skills/ml-experiment-loop/scripts/log_experiment.py`
自动追加。

---

## 10. （可选）在我们 preprocessor 下重新 Optuna tune

如果 §5 跑出来 single Transformer 分数比你原 repo 那边低不少（说明
feature encoding 差异影响了最优超参），可以在我们这边重新 tune：

```bash
python main.py --mode tune --task classification --model tabular_transformer \
    --sampler tpe --pruner hyperband
```

默认 60 trials，3-fold inner CV。best_params 写到 `outputs/metrics/`
下的 JSON。搜索空间在 `configs/config.py` →
`tabular_transformer_clf_search_space`（已按你 IDEAS.md 的范围设好）。

tune 完再跑 5-fold CV：

```bash
python main.py --mode cv --task classification --model tabular_transformer \
    --apply_tune_results outputs/metrics/<那个tune输出的json>
```

---

## 11. 建议的第一批实验（按优先级）

对应 §4-§8 的主路径：

1. **先应用你的 best_params**（§4，5 分钟动手）
2. **cls single 5-fold CV**（§5，~30 分钟）—— 看分数 & 攒 Transformer OOF。
   判据：mean ≥ 0.855 → §8.1 值得做；< 0.845 → 先跑 §10 tune。
3. **复现我这边 cls SOTA**（§6，~7 小时 cls 部分）—— 攒 3 份 baseline OOF。
4. **Error analysis**（§7，几分钟）—— 看 Q(transformer, tabnet)：
   - `< 0.90` → §8.1 combined stacking 大概率值得做
   - `≥ 0.94` → 先回 §10 重 tune，不然 stacking 必稀释
5. **reg single 5-fold CV**（§5）
6. **cls combined stacking**（§8.1，~3 小时）—— **对标 0.8679**。
7. **cls gated 扫描**（§8.2，~5 分钟 OOF 计算）—— **对标 0.8718**。
8. **reg combined stacking**（§8.4，~3 小时）—— 对标 0.8406。
9. （可选）**tune in our preprocessor**（§10）
10. （可选）**全训 + 提交**（§8.3）

---

## 12. 常见坑

- **CUDA 没用上**: 检查 `torch.cuda.is_available()`。wrapper 的
  `_pick_device` 会自动选 CUDA。强制指定可以在 params 里加
  `"device": "cuda"`。
- **MPS NaN**: 我在 `_pick_device` 里故意跳过 MPS —— torch 2.11 在 macOS
  上会 NaN。你 4090 不会碰到，CUDA 正常。
- **Embedding index out of range**: 如果你改了 preprocessor 的 cat
  encoding，`cat_dims` 会变。wrapper 自动加 +2 slack，但如果 test 集有
  比 train 多 5+ 的新类别，需要手动加大。
- **NaN loss**: `lr` 从默认降到 3e-4，或加大 `grad_clip` 到 2.0。
- **OOM on 4090**: `batch_size` 从 256 降到 128 或 64。17 个 TabNet+CORN-MLP
  同时 fit 会占不少显存；单独一个 Transformer 应该完全不紧。
- **Stacking 微降而非微涨**: 先检查 §5 single 分数。Row #46-47 证明
  single 弱于 TabNet single > 1pt 塞进来必稀释。改用小 `n_transformer`（1–2）再试。

---

## 13. 代码入口一览

| 文件 | 作用 |
|------|------|
| `modeling/_tabular_transformer.py` | 纯 nn.Module backbone（从你 repo 搬过来的） |
| `modeling/_ordinal_heads.py` | CORN loss / decode（无外部依赖） |
| `modeling/tabular_transformer_model.py` | BaseModel 包装，fit/predict/predict_proba |
| `modeling/coral_mlp_model.py` | 5 MLP 那一档（CORN-MLP） |
| `modeling/tabnet_model.py` | 12 TabNet 那一档 + gate 用的 single |
| `configs/config.py` | 超参 + 搜索空间 + ensemble generators |
| `training/cross_validator.py` | 5-fold CV 调度器（自动 OOF + error analysis） |
| `modeling/ensemble_model.py` | Stacking / uniform / weighted ensemble |
| `analysis/error_analyzer.py` | Confusion matrix + hard examples + confidence bins |
| `analysis/error_correlation.py` | **跨模型 Yule's Q / 错题 overlap**（§7.3 入口） |
| `scripts/stack_tabnet_coral.py` | 当前 SOTA 的 stacking CV 驱动（§6.2） |
| `scripts/generate_gated_submission.py` | **当前 SOTA 的全训+提交脚本**（改它跑新 SOTA） |
| `scripts/two_stage_v2_with_sota_oof.py` | **5 步编排：一条命令复现 SOTA**（§6.1） |
| `scripts/smoke_tabular_transformer.py` | 本地 wiring 验证 |
| `experiments.md` | 所有实验日志 + Phase 级反思 |

---

## 14. 遇到问题

- `experiments.md` 的 Reflection 章节记着我踩过的坑（tree 家族
  Q=0.96 高相关、FT-Transformer CPU 不收敛、单模型弱于 TabNet
  single > 1pt 就别进 stacking 等）。
- 我当前 SOTA 路径: row #30 → #41 → #46 → #47 → #48 → #49 → #51。
  重点读 #47 (stack base 0.8679) + #48 (gated 0.8718) 这两行。
- 有问题直接 issue 或者 DM。

祝 4090 跑出新 SOTA。
