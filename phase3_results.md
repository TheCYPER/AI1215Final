# Phase III — 新基模型实验结果

Worktree: `/Users/percy/MLFinal2026-phase3` (branch `phase3-base-models`).
对比 baseline: row #14 tuned CatBoost **0.8263 ± 0.0023**; row #15 stacking **0.8331 ± 0.0016**.

## 全部结果汇总

| id | 时间 | task | model | 假设 / 改动 | 新分数 | delta vs row#15 SOTA | 结论 | 备注 |
|----|------|------|-------|------------|--------|---------------------|------|------|
| P3-1 | 04-16 00:38 | cls | LogRegPoly(2) | degree=2 interaction_only + C=0.1；线性 baseline | 0.7152 ± 0.0054 | −0.1179 | expected | 线性模型天花板低；ensemble diversity member |
| P3-2 | 04-16 00:39 | cls | MLP(256,128) BROKEN | sklearn MLPClassifier 默认参数；cat 列未 scale → 梯度爆炸 | 0.3650 ± 0.0107 | — | BUG | 已修复：MLP.fit 加 StandardScaler |
| P3-3 | 04-16 10:16 | cls | **MLP(256,128) fixed baseline** | 加 StandardScaler 后重跑 | **0.8046 ± 0.0062** | −0.0285 | OK | 仅次于 CatBoost 的第二强单模型 |
| P3-4 | 04-16 10:58 | cls | **MLP Optuna-tuned** | TPE 60 trials; best: hidden=(158,79), alpha=0.0898, lr=0.00855, batch=66, max_iter=400 | **0.8145 ± 0.0061** | −0.0186 | **+0.0099 vs baseline** | tune 提升 ~1 pt；MLP 在 tree 之外的最强异质模型 |
| P3-5 | 04-16 01:40 | cls | LGBM widened-tuned | TPE 60 trials widened space; best: n_est=1784, lr=0.024, num_leaves=32, reg_lambda=95.5 | 0.7928 ± 0.0025 | −0.0403 | at ceiling | 极高正则（reg_lambda=95.5）；和 default LGBM 持平 |
| P3-6 | 04-16 02:16 | cls | XGB widened-tuned | TPE 60 trials widened space; best: n_est=1942, lr=0.023, reg_lambda=56.6 | 0.7875 ± 0.0023 | −0.0456 | at ceiling | 极高正则（reg_lambda=56.6）；和 default XGB 持平 |
| P3-7 | 04-16 10:22 | cls | **TabNet baseline** | pytorch-tabnet 默认参数 (n_d=16, n_a=16, n_steps=4, patience=15) | **0.8512 ± 0.0090** | **+0.0181** 🔥🔥 | **NEW SOTA** | 默认 params 碾压一切！std 高（NN 特性）但最低折 0.8337 仍 ≥ stacking |
| P3-8 | running | cls | TabNet Optuna tune | TPE 60 trials in progress | pending | | | 搜索 n_d/n_a/n_steps/gamma/lambda_sparse/max_epochs/patience |

## 完整模型排名

| 排名 | 模型 | 5-fold acc | ± std | 类型 |
|------|------|-----------|-------|------|
| 1 | **TabNet baseline** | **0.8512** | 0.0090 | NN (attention) |
| 2 | 30-CatBoost stacking | 0.8331 | 0.0016 | tree ensemble+meta |
| 3 | CatBoost TPE-tuned | 0.8263 | 0.0023 | tree |
| 4 | CatBoost default | 0.8234 | 0.0038 | tree |
| 5 | **MLP tuned** | **0.8145** | 0.0061 | NN (feedforward) |
| 6 | MLP baseline | 0.8046 | 0.0062 | NN (feedforward) |
| 7 | LGBM widened-tuned | 0.7928 | 0.0025 | tree |
| 8 | XGB widened-tuned | 0.7875 | 0.0023 | tree |
| 9 | LogRegPoly | 0.7152 | 0.0054 | linear |

## 待办

- [ ] TabNet Optuna tune 完成 → apply best → 5-fold CV → 可能新 SOTA 0.86+
- [ ] TabNet 加速：切 MPS (GPU) + OMP_NUM_THREADS 4-8（Optuna 完成后改）
- [ ] TabNet + CatBoost + MLP 三异质模型 stacking
- [ ] 最终 submission 生成
