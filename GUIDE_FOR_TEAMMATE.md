# 跑 TabularTransformer 实验的指南（给队友）

Percy 版 repo: <https://github.com/TheCYPER/AI1215Final>

你仓库里的 `TabularTransformer` 已经被搬到这个 repo，注册进了我们的
`BaseModel` 工厂。现在它跟 CatBoost / TabNet / CORN-MLP 共享同一套
CV / stacking / tuning / submission 管线。以下指南假定你用
CUDA 环境继续跑真正的实验。

---

## 0. 主要思路：把两边 SOTA 合起来跑 Ensemble

现在双方各有自己的 SOTA：

- **我这边 SOTA**: 12-TabNet stacking (cls 0.8687 / reg 0.8406)
- **你那边 SOTA**: 你调好的单个 / stacking TabularTransformer

两边架构完全不同 —— TabNet 是 sparse-attention 的 sequential decision
process，Transformer 是 dense self-attention over feature tokens ——
归纳偏置正交。按 `experiments.md` row #40 的错题分析结论，这种跨架构
组合是唯一还没被我耗尽的破 SOTA 路径。

**主目标**: 把你的 Transformer bases 塞进我这边的 12-TabNet stacking，
看能不能把 cls 推过 0.8687 / reg 推过 0.8406。

**次要目标**（有时间再做）: 你自己的 single Transformer / pure
Transformer stacking 在我们 preprocessor 下跑多少。

具体流程：

1. 把你原 repo 的 Optuna 最佳超参搬到我们 `configs/config.py`
2. 跑 5-fold CV 生成 Transformer 的 OOF
3. 写一个 generator 把 `12 TabNet + N Transformer` 塞进 stacking
4. 跑 ensemble CV，对标我们的 SOTA

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
让你确认一下拉下来的 repo 能跑。

---

## 3. 把你的 Transformer SOTA 超参应用到这边

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
    "use_ordinal": True,           # CORN 头（建议开；破 softmax ~1pt）
    "max_epochs": 100, "patience": 15,
    "grad_clip": 1.0, "seed": 42,
})
```

回归同理，改 `tabular_transformer_reg_params`。

**选项 B**: 重新跑 Optuna（耗时；只在想要我们 preprocessor 下的重新
tune 时用）。见 §7。

---

## 4. 跑单模型 CV 生成 OOF

超参改好后跑一次 5-fold CV —— 目的有两个：
(a) 看 single Transformer 在我们的 preprocessor 下的分数；
(b) 保存 OOF，后面 stacking 要用。

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
- `outputs/oof/tabular_transformer.npz` ← OOF preds（stacking 要用）
- `outputs/analysis/` ← cls 自动跑的错题分析

---

## 5. 对标现有 SOTA

| 任务 | 当前 SOTA | 来源 |
|------|-----------|------|
| cls  | 0.8687 ± 0.0040 | 12-TabNet stacking (row #30) |
| cls (gated OOF) | 0.8718 | stack_12+5 + TabNet-single gating @ t=0.59 (row #48) |
| reg  | 0.8406 ± 0.0152 | 12-TabNet reg stacking (row #49) |

对照 single-model baselines（用来判断你 Transformer single 值不值得进 ensemble）：

| 任务 | TabNet single | CORN-MLP | CatBoost |
|------|---------------|----------|----------|
| cls  | 0.8573 ± 0.0049 | 0.8373 ± 0.0028 | 0.8263 ± 0.0023 |
| reg  | 0.8272 ± 0.0242 (5-fold) | — | 0.8367 ± 0.0148 |

**经验判据（来自 experiments.md row #46-47）**: 单模型比 TabNet single
(0.8573) 差 > 1pt 时，塞进 12-TabNet stacking 通常会稀释而不是帮助。
所以 §4 跑出来 < 0.845 的话，§6 的 combined stacking 先别抱太大期望，
可以考虑先回 §7 tune。

---

## 6. 核心实验：12 TabNet + N TabularTransformer stacking

**这是主要路径。** 写一个 generator 把两边的 SOTA bases 拼起来：

在 `configs/config.py` 文件内（跟其他 generator 放一块，比如
`pure_tabnet_stacking_components` 旁边）加：

```python
def tabnet_plus_transformer_components(
    n_transformer: int = 3,
) -> List[Dict[str, Any]]:
    """我的 12-TabNet SOTA + N 个 seed-jittered TabularTransformer.

    假设：Transformer 的 dense self-attention 归纳偏置与 TabNet 的
    sparse sequential attention 正交（row #40 错题分析说明需要换架构
    才能真正 decorrelate），meta learner 能在 (12+N)*5=... 维 OOF
    proba 上找到利用方式。
    """
    components = pure_tabnet_stacking_components()  # 12 TabNet
    for i in range(n_transformer):
        overrides = {
            "seed": 500 + i,
            # 让 N 个 variant 走不同 numerical embedding，放大架构内多样性：
            "num_embedding_type": ["numerical", "periodic", "ple"][i % 3],
        }
        components.append({
            "type": "tabular_transformer",
            "overrides": overrides,
        })
    return components
```

然后让 `ensemble_clf_components` 指过去：

```python
# configs/config.py, class ModelConfig
ensemble_clf_components: List[Any] = field(
    default_factory=lambda: tabnet_plus_transformer_components(n_transformer=3)
)
```

跑：

```bash
python main.py --mode cv --task classification --model ensemble
```

这会：

- 对每个 fold 训练 12 个 TabNet + 3 个 Transformer，共 15 个 base；
- 在 80/20 holdout 上训 LogReg meta（75 维 OOF proba feature）；
- 汇总得 5-fold mean ± std。

4090 上预计 **90–180 分钟**（TabNet 15 个 + Transformer 3 个）。
结果对标 SOTA 0.8687。

### 回归版本

`pure_tabnet_reg_stacking_components` 是我们的 12-TabNet reg stack
(row #49, r² 0.8406)。写个对应的回归 generator：

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

```python
ensemble_reg_components: List[Any] = field(
    default_factory=lambda: tabnet_plus_transformer_reg_components(n_transformer=3)
)
ensemble_reg_mode: str = "stacking"  # 默认是 uniform，记得改
ensemble_meta_learner_type: str = "ridge"  # reg 用 Ridge
```

跑：

```bash
python main.py --mode cv --task regression --model ensemble
```

对标 reg SOTA 0.8406。

### N 的扫描

第一轮推荐 `n_transformer=3`（15 bases 总量接近 row #30 的 12，便于对比）。
如果 +3 拉过 SOTA 了，可以再试 5 / 7 看边际收益；如果 +3 微降，试 1 / 2
看稀释规律。这个是 row #46-47 对 CORN-MLP 做过的 ablation。

---

## 7. （可选）在我们 preprocessor 下重新 Optuna tune

如果 §4 跑出来 single Transformer 分数比你原 repo 那边低不少（说明
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

## 8. （可选）Gated mixture 路径

Row #48 证明过：把 TabNet single 当 low-confidence specialist 挂在
stacking 后面 (stack @ threshold 0.59 → TabNet single) 能再 +0.003。
如果 §6 的 combined stacking 出了新 cls 分数，可以用同模板把
Transformer single 当 specialist 试一下：

```
Gated(combined_stack, Transformer_single) @ threshold=?
```

OOF 级扫描脚本参考 `/tmp/test_gated_stacking.py`（session 临时文件，
需要我这边 dm 给你）。

---

## 9. 建议的第一批实验（按优先级）

对应 §3-§6 的主路径：

1. **先应用你的 best_params**（§3，5 分钟动手）
2. **cls single 5-fold CV**（§4，~30 分钟）—— 看分数 & 攒 OOF。
   判据：mean ≥ 0.845 → combined stacking 值得做；低于 0.840 → 先跑 §7 tune。
3. **reg single 5-fold CV**（§4，~30 分钟）—— 同上攒 OOF。
4. **cls combined stacking**（§6，~2-3 小时）—— **破 0.8687 的主攻方向**。
5. **reg combined stacking**（§6，~2-3 小时）—— 对标 0.8406。
6. （可选）**tune in our preprocessor**（§7）
7. （可选）**gated 混合**（§8）

---

## 10. 实验记录

每跑完一个 CV，在 `experiments.md` 末尾加一行。表格的 8 列：

| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 |

**结论判准**（`ml-experiment-loop` skill 的默认定义）：

- `work`: delta > 2 × baseline std
- `待定`: |delta| < 1 × baseline std
- `not work`: delta < -2 × baseline std

也可以用 `python .claude/skills/ml-experiment-loop/scripts/log_experiment.py`
自动追加。

---

## 11. 常见坑

- **CUDA 没用上**: 检查 `torch.cuda.is_available()`。wrapper 的
  `_pick_device` 会自动选 CUDA。强制指定可以在 params 里加
  `"device": "cuda"`。
- **MPS NaN**: 我在 `_pick_device` 里故意跳过 MPS —— torch 2.11 在 macOS
  上会 NaN。你 4090 不会碰到，CUDA 正常。
- **Embedding index out of range**: 如果你改了 preprocessor 的 cat
  encoding，`cat_dims` 会变。wrapper 自动加 +2 slack，但如果 test 集有
  比 train 多 5+ 的新类别，需要手动加大。
- **NaN loss**: `lr` 从默认降到 3e-4，或加大 `grad_clip` 到 2.0。
- **OOM on 4090**: `batch_size` 从 256 降到 128 或 64。
- **Stacking 微降而非微涨**: 先检查 §4 的 single 分数。Row #46-47 证明
  single 弱 > 1pt 塞进来必稀释。改用小 `n_transformer`（1–2）再试。

---

## 12. 代码入口一览

| 文件 | 作用 |
|------|------|
| `modeling/_tabular_transformer.py` | 纯 nn.Module backbone（从你 repo 搬过来的） |
| `modeling/_ordinal_heads.py` | CORN loss / decode（无外部依赖） |
| `modeling/tabular_transformer_model.py` | BaseModel 包装，fit/predict/predict_proba |
| `configs/config.py` | 超参 + 搜索空间 + ensemble generators |
| `training/cross_validator.py` | 5-fold CV 调度器 |
| `modeling/ensemble_model.py` | Stacking / uniform / weighted ensemble |
| `scripts/smoke_tabular_transformer.py` | 本地 wiring 验证 |
| `experiments.md` | 所有实验日志 + Phase 级反思 |

---

## 13. 遇到问题

- `experiments.md` 的 Reflection 章节记着我踩过的坑（tree 家族
  Q=0.96 高相关、FT-Transformer CPU 不收敛、单模型弱于 TabNet
  single > 1pt 就别进 stacking 等）。
- 我当前 SOTA 路径：row #30 → #41 → #46 → #48 → #49 → #51。
- 有问题直接 issue 或者 DM。

祝 4090 跑出新 SOTA。
