#!/usr/bin/env python3
"""Append one experiment row to experiments.md.

Reads the latest metrics produced by `python main.py --mode cv` and the last
row of experiments.md to compute a delta automatically. Leaves the 结论 column
blank so the caller (Claude) can judge against CV std.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[4]
EXPERIMENTS_MD = PROJECT_ROOT / "experiments.md"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
TEMPLATE = SCRIPT.parents[1] / "assets" / "experiments_template.md"

TASK_SHORT = {"classification": "cls", "regression": "reg"}


def read_latest_score(task):
    """Return (score_str, score_num, std_or_none) for the task's primary metric."""
    if task == "classification":
        cv = METRICS_DIR / "classification_cv_summary.json"
        if cv.exists():
            d = json.loads(cv.read_text())
            mean, std = d["accuracy_mean"], d.get("accuracy_std", 0.0)
            return f"{mean:.4f} ± {std:.4f}", mean, std
        single = METRICS_DIR / "classification_metrics.json"
        if single.exists():
            d = json.loads(single.read_text())
            acc = d["val_metrics"]["accuracy"]
            return f"{acc:.4f} (val)", acc, None
    elif task == "regression":
        single = METRICS_DIR / "regression_metrics.json"
        if single.exists():
            d = json.loads(single.read_text())
            r2 = d["val_metrics"]["r2"]
            rmse = d["val_metrics"]["rmse"]
            return f"r2={r2:.4f} rmse={rmse:.4f}", r2, None
    sys.exit(
        f"no metrics for task={task} under {METRICS_DIR}. "
        f"run `python main.py --mode cv --task {task}` first."
    )


def read_last_baseline(task):
    """Parse experiments.md for the latest score on this task. Returns (str, num, next_id)."""
    if not EXPERIMENTS_MD.exists():
        return None, None, 0

    short = TASK_SHORT[task]
    max_id = -1
    last_str, last_num = None, None

    for line in EXPERIMENTS_MD.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        try:
            rid = int(cells[0])
        except ValueError:
            continue
        max_id = max(max_id, rid)
        if cells[2] == short:
            m = re.search(r"[-+]?\d*\.?\d+", cells[6])
            if m:
                last_num = float(m.group(0))
                last_str = cells[6]

    return last_str, last_num, max_id + 1


def truncate(s, n=60):
    s = s.replace("|", "\\|").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def ensure_log_exists():
    if EXPERIMENTS_MD.exists():
        return
    if TEMPLATE.exists():
        EXPERIMENTS_MD.write_text(TEMPLATE.read_text())
    else:
        EXPERIMENTS_MD.write_text(
            "# CreditSense Experiments Log\n\n"
            "| id | 时间 | task | 假设 | 改动点 | baseline | 新分数 | delta | 结论 | 备注 |\n"
            "|----|------|------|------|--------|----------|--------|-------|------|------|\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--baseline-override", default=None,
                   help="manual baseline, e.g. '0.7850 ± 0.0025' — for the first row")
    args = p.parse_args()

    new_str, new_num, _ = read_latest_score(args.task)
    base_str, base_num, next_id = read_last_baseline(args.task)

    if args.baseline_override:
        base_str = args.baseline_override
        m = re.search(r"[-+]?\d*\.?\d+", args.baseline_override)
        base_num = float(m.group(0)) if m else None

    if base_num is not None:
        delta_str = f"{new_num - base_num:+.4f}"
    else:
        base_str, delta_str = "—", "—"

    ensure_log_exists()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {next_id} | {ts} | {TASK_SHORT[args.task]} "
        f"| {truncate(args.hypothesis)} | {truncate(args.change)} "
        f"| {base_str} | {new_str} | {delta_str} |  | {truncate(args.notes, 50)} |"
    )
    with EXPERIMENTS_MD.open("a") as f:
        f.write(row + "\n")

    print(f"logged row #{next_id} ({args.task})")
    print(f"  baseline: {base_str}")
    print(f"  new:      {new_str}")
    print(f"  delta:    {delta_str}")
    print("→ now fill in 结论 column based on delta vs CV std (see SKILL.md step 5)")


if __name__ == "__main__":
    main()
