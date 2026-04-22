# 跑 TabularTransformer 实验的指南（给队友）

Percy 版 repo: <https://github.com/TheCYPER/AI1215Final>

你仓库里的 `TabularTransformer` 已经被搬到这个 repo，注册进了我们的
`BaseModel` 工厂。现在它跟 CatBoost / TabNet / CORN-MLP 共享同一套
CV / stacking / tuning / submission 管线。以下指南假定你用
CUDA 环境继续跑真正的实验。

---

## 1. 拉代码 + 环境

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
data/sample_submission.csv   # 可选，/mode submit 时要
```

验证 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. 先跑 smoke test

纯 wiring 检查，CPU 一秒过：

```bash
python scripts/smoke_tabular_transformer.py
# all smoke checks passed
```

这只验证 fit / predict / predict_proba 形状和数值正常 —— 不看分数。

---

## 3. 跑真正的 5-fold CV

### 单模型 Transformer（分类）

```bash
python main.py --mode cv --task classification --model tabular_transformer
```

默认参数（`configs/config.py` → `tabular_transformer_clf_params`）：

```python
{
    "d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 256,
    "dropout": 0.15, "num_embedding_type": "numerical",
    "use_cls_token": True, "pooling": "cls", "pre_norm": True,
    "use_ordinal": True,            # CORN 头，不是 softmax
    "max_epochs": 100, "patience": 15, "batch_size": 256,
    "lr": 9e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
}
```

4090 上一次 5-fold 估计 **15–30 分钟**。

输出：

- `outputs/metrics/classification_cv_summary.json` ← fold mean / std
- `outputs/oof/tabular_transformer.npz` ← OOF preds（后面 stacking 用）
- `outputs/analysis/` ← 自动跑的错误分析

### 单模型 Transformer（回归）

```bash
python main.py --mode cv --task regression --model tabular_transformer
```

同样 5-fold，CV 时间类似。回归目标会在 wrapper 内部标准化，你拿到的
`r2` 是真实 InterestRate 尺度上算的。

---

## 4. 对标现有 SOTA

| 任务 | 当前 SOTA | 来源 |
|------|-----------|------|
| cls  | 0.8687 ± 0.0040 | 12-TabNet stacking (row #30) |
| cls (gated OOF) | 0.8718 | stack_12+5 + TabNet-single gating @ t=0.59 (row #48) |
| reg  | 0.8406 ± 0.0152 | 12-TabNet reg stacking (row #49) |

对照 single-model baselines：

| 任务 | TabNet single | CORN-MLP | CatBoost |
|------|---------------|----------|----------|
| cls  | 0.8573 ± 0.0049 | 0.8373 ± 0.0028 | 0.8263 ± 0.0023 |
| reg  | 0.8272 ± 0.0242 (5-fold) | — | 0.8367 ± 0.0148 |

你第一个目标：**看 single TabularTransformer 能不能过这些 single baseline**。

---

## 5. 改超参

所有模型参数集中在 `configs/config.py` 的 `ModelConfig` dataclass。
不要去碰 `modeling/` 下的源代码改超参数。

想临时覆盖，直接编辑 `configs/config.py`：

```python
tabular_transformer_clf_params = {
    ...,
    "use_ordinal": False,         # 关闭 CORN，退回 5-way CE
    "num_embedding_type": "ple",  # 换成 piecewise-linear 嵌入
}
```

几个值得试的开关：

- `num_embedding_type`: `"numerical"` / `"periodic"` / `"ple"` / `"linear"`
- `use_cls_token`: True / False（False 会走 mean-pool）
- `use_column_embedding`: True（给每个特征加列身份 + 类型嵌入）
- `use_ordinal`: True（CORN）/ False（softmax CE）

---

## 6. Optuna tune

搜索空间已经按你 `IDEAS.md` 的范围设好了，跑就完事：

```bash
python main.py --mode tune --task classification --model tabular_transformer \
    --sampler tpe --pruner hyperband
```

默认 60 trials，3-fold inner CV。best_params 自动写到
`outputs/metrics/` 下的 JSON，再拿回去跑 5-fold CV：

```bash
python main.py --mode cv --task classification --model tabular_transformer \
    --apply_tune_results outputs/metrics/<那个tune输出的json>
```

搜索空间在 `configs/config.py` → `tabular_transformer_clf_search_space`。
如果觉得范围不够宽，改那里。

---

## 7. 把 Transformer 塞进 stacking

**这是最有可能破 SOTA 的方向。** 现在 cls SOTA 是 12 个 TabNet
stacking；加 N 个 architectural-different 的 Transformer 进去，理论上
应该比单加 CORN-MLP（row #46-47 证实微降）有更真的架构多样性 ——
self-attention + feature-token 和 TabNet 的 sparse attention 是两个
完全不同的归纳偏置。

新写一个生成器（在 `configs/config.py` 里）：

```python
def tabnet_plus_transformer_components():
    """12 TabNet + 3 TabularTransformer (seed-jittered)."""
    components = pure_tabnet_stacking_components()  # 12 TabNet
    for seed in range(3):
        components.append({
            "type": "tabular_transformer",
            "overrides": {
                "seed": 400 + seed,
                "max_epochs": 80,
                "patience": 12,
                # 可选：让 3 个 variant 用不同嵌入，增加多样性
                "num_embedding_type": ["numerical", "periodic", "ple"][seed],
            },
        })
    return components
```

然后把 `ensemble_clf_components` 指过去：

```python
ensemble_clf_components: List[Any] = field(
    default_factory=lambda: tabnet_plus_transformer_components()
)
```

跑：

```bash
python main.py --mode cv --task classification --model ensemble
```

Stacking meta learner（LogReg）会在 15 × 5 = 75 维 OOF proba 特征上
学非线性组合。CV 下来 15–18 min/fold × 5 = 2 小时左右。

---

## 8. 实验记录

每跑完一个 CV，在 `experiments.md` 末尾加一行。表格的 8 列：

| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 |

**结论判准**（这个 repo 的 `ml-experiment-loop` skill 里定义）：

- `work`: delta > 2 × baseline std
- `待定`: |delta| < 1 × baseline std
- `not work`: delta < -2 × baseline std

也可以用 `python .claude/skills/ml-experiment-loop/scripts/log_experiment.py`
自动追加。

---

## 9. 建议的第一批实验（按优先级）

1. **cls single baseline**: `--mode cv --task classification --model tabular_transformer`
   默认 params。对标 TabNet single 0.8573 / SOTA 0.8687。
2. **reg single baseline**: `--mode cv --task regression --model tabular_transformer`
   对标 TabNet reg 0.8272 / CatBoost 0.8367 / reg stack 0.8406。
3. **cls Optuna tune**: 60 trials TPE，应用后重跑 5-fold。
4. **cls stacking**: 12 TabNet + 3 TabularTransformer（第 7 节）。
   **这是最重要的一个** —— 破 0.8687 的主攻方向。
5. **reg stacking**: 12 TabNet reg + 3 TabularTransformer reg。对标 0.8406。
6. （可选）**Two-stage**: 把 cls SOTA OOF 塞进 reg feature（row #51 模板），
   看 Transformer 回归有没有比 TabNet 更吃这个 prior。

---

## 10. 常见坑

- **CUDA 没用上**: 检查 `torch.cuda.is_available()`。我 wrapper 的
  `_pick_device` 会自动选 CUDA（没有就退 CPU）。强制指定可以在
  params 里加 `"device": "cuda"`。
- **MPS NaN**: 我在 `_pick_device` 里故意跳过 MPS —— torch 2.11 在 macOS
  上会 NaN。你 4090 不会碰到，CUDA 正常。
- **Embedding index out of range**: 如果你改了 preprocessor 的 cat
  encoding，`cat_dims` 会变。wrapper 自动加 +2 slack，但如果 test 集有
  比 train 多 5+ 的新类别，需要手动加大。
- **NaN loss**: `lr` 从 9e-4 降到 3e-4，或加大 `grad_clip` 到 2.0。
- **OOM on 4090**: `batch_size` 从 256 降到 128 或 64。

---

## 11. 代码入口一览

| 文件 | 作用 |
|------|------|
| `modeling/_tabular_transformer.py` | 纯 nn.Module backbone（从你 repo 搬过来的） |
| `modeling/_ordinal_heads.py` | CORN loss / decode（无外部依赖） |
| `modeling/tabular_transformer_model.py` | BaseModel 包装，fit/predict/predict_proba |
| `configs/config.py` | 超参 + 搜索空间（`tabular_transformer_*`） |
| `training/cross_validator.py` | 5-fold CV 调度器 |
| `modeling/ensemble_model.py` | Stacking / uniform / weighted ensemble |
| `scripts/smoke_tabular_transformer.py` | 本地 wiring 验证 |
| `experiments.md` | 所有实验日志 + Phase 级反思 |

---

## 12. 遇到问题

- `experiments.md` 的 Reflection 章节记着我踩过的坑（tree 家族
  Q=0.96 高相关、FT-Transformer CPU 不收敛等）。
- 我当前 SOTA 路径：row #30 → #41 → #46 → #48 → #49 → #51。
- 有问题直接 issue 或者 DM。
