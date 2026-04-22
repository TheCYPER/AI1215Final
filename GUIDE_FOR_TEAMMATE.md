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
> 1pt，塞进 stacking 通常稀释而非帮助。Single ≥ 0.855 → §6 combined
stacking 值得跑；< 0.845 → 先回 §8 tune。

---

## 6. 核心实验：把 Transformer 加进 SOTA

### 6.1 加进 stacking base（第一优先级）

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

### 6.2 再挂上 gate → 对标 0.8718

跑完 §6.1 CV 后，`outputs/oof/` 下会有新 stack OOF 文件。用这个 OOF
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

### 6.3 全数据训练 + Kaggle 提交

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

### 6.4 回归版本

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

### 6.5 N 的扫描

第一轮推荐 `n_transformer=3`。如果 +3 拉过 SOTA 了，可以再试 5 / 7
看边际收益；如果 +3 微降，试 1 / 2 看稀释规律。这个是 row #46-47 对
CORN-MLP 做过的 ablation（3 → 5 CORN-MLP 只 +0.0004）。

---

## 7. 实验记录

每跑完一个 CV，在 `experiments.md` 末尾加一行。表格的 9 列：

| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 | 备注 |

**结论判准**（`ml-experiment-loop` skill 的默认定义）：

- `work`: delta > 2 × baseline std
- `待定`: |delta| < 1 × baseline std
- `not work`: delta < -2 × baseline std

也可以用 `python .claude/skills/ml-experiment-loop/scripts/log_experiment.py`
自动追加。

---

## 8. （可选）在我们 preprocessor 下重新 Optuna tune

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

## 9. 建议的第一批实验（按优先级）

对应 §4-§6 的主路径：

1. **先应用你的 best_params**（§4，5 分钟动手）
2. **cls single 5-fold CV**（§5，~30 分钟）—— 看分数 & 攒 OOF。
   判据：mean ≥ 0.855 → §6.1 值得做；< 0.845 → 先跑 §8 tune。
3. **reg single 5-fold CV**（§5，~30 分钟）—— 同上攒 OOF。
4. **cls combined stacking**（§6.1，~3 小时）—— **对标 0.8679**。
5. **cls gated 扫描**（§6.2，~5 分钟 OOF 计算）—— **对标 0.8718**。
6. **reg combined stacking**（§6.4，~3 小时）—— 对标 0.8406。
7. （可选）**tune in our preprocessor**（§8）
8. （可选）**全训 + 提交**（§6.3）

---

## 10. 常见坑

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

## 11. 代码入口一览

| 文件 | 作用 |
|------|------|
| `modeling/_tabular_transformer.py` | 纯 nn.Module backbone（从你 repo 搬过来的） |
| `modeling/_ordinal_heads.py` | CORN loss / decode（无外部依赖） |
| `modeling/tabular_transformer_model.py` | BaseModel 包装，fit/predict/predict_proba |
| `modeling/coral_mlp_model.py` | 5 MLP 那一档（CORN-MLP） |
| `modeling/tabnet_model.py` | 12 TabNet 那一档 + gate 用的 single |
| `configs/config.py` | 超参 + 搜索空间 + ensemble generators |
| `training/cross_validator.py` | 5-fold CV 调度器 |
| `modeling/ensemble_model.py` | Stacking / uniform / weighted ensemble |
| `scripts/stack_tabnet_coral.py` | 当前 SOTA 的 stacking CV 驱动 |
| `scripts/generate_gated_submission.py` | **当前 SOTA 的全训+提交脚本**（改它跑新 SOTA） |
| `scripts/two_stage_v2_with_sota_oof.py` | 5 步编排：重生成 gated OOF（ref 实现） |
| `scripts/smoke_tabular_transformer.py` | 本地 wiring 验证 |
| `experiments.md` | 所有实验日志 + Phase 级反思 |

---

## 12. 遇到问题

- `experiments.md` 的 Reflection 章节记着我踩过的坑（tree 家族
  Q=0.96 高相关、FT-Transformer CPU 不收敛、单模型弱于 TabNet
  single > 1pt 就别进 stacking 等）。
- 我当前 SOTA 路径: row #30 → #41 → #46 → #47 → #48 → #49 → #51。
  重点读 #47 (stack base 0.8679) + #48 (gated 0.8718) 这两行。
- 有问题直接 issue 或者 DM。

祝 4090 跑出新 SOTA。
