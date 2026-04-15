# CreditSense Experiments Log

每行一个实验。跑完 CV 后用 `log_experiment.py` 追加，结论列由人/Claude 填。

列说明：
- **baseline**: 该 task 上一行的「新分数」
- **delta**: 新分数 − baseline
- **结论**: `work` (delta > 2×std) / `待定` (|delta| < 1×std) / `not work` (delta < −2×std)

| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 | 备注 |
|----|------|------|------|--------|----------|--------|-------|------|------|
