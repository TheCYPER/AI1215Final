"""
Exploratory Data Analysis — generates a visual report for all columns.

Outputs PNG charts to outputs/eda/:
  - Numeric columns: histogram + boxplot by RiskTier
  - Categorical columns: count bar chart + stacked proportion by RiskTier
  - Correlation heatmap
  - Target distribution plots
  - Missing value summary
"""

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.config import Config

logger = logging.getLogger("eda")

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
FIGSIZE_SINGLE = (8, 5)
FIGSIZE_DOUBLE = (14, 5)


def run_eda(config: Config) -> None:
    """Run full EDA and save all plots."""
    out_dir = Path(config.paths.output_dir) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.paths.train_csv)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    target_clf = config.columns.classification_target
    target_reg = config.columns.regression_target

    # Separate column types
    cat_cols = (
        config.columns.categorical
        + config.columns.forced_categorical
    )
    num_cols = [
        c for c in df.columns
        if c not in cat_cols
        and c not in config.columns.targets
        and df[c].dtype != "object"
    ]

    # 1. Missing values
    _plot_missing(df, out_dir)

    # 2. Target distributions
    _plot_target_clf(df, target_clf, out_dir)
    _plot_target_reg(df, target_reg, out_dir)

    # 3. Numeric columns
    for col in num_cols:
        _plot_numeric(df, col, target_clf, out_dir)

    # 4. Categorical columns
    for col in cat_cols:
        if col in df.columns:
            _plot_categorical(df, col, target_clf, out_dir)

    # 5. Correlation heatmap (top features)
    _plot_correlation(df, num_cols, out_dir)

    logger.info(f"EDA complete — {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}")


def _plot_missing(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of missing value percentages."""
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        logger.info("No missing values found")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.3)))
    missing.plot.barh(ax=ax, color="salmon")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    ax.invert_yaxis()
    for i, v in enumerate(missing):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "00_missing_values.png", dpi=150)
    plt.close(fig)


def _plot_target_clf(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """RiskTier distribution."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    counts = df[target].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("coolwarm", 5))
    ax.set_xlabel("RiskTier")
    ax.set_ylabel("Count")
    ax.set_title(f"Target Distribution: {target}")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "01_target_risktier.png", dpi=150)
    plt.close(fig)


def _plot_target_reg(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """InterestRate distribution."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    df[target].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title(f"{target} — Histogram")
    axes[0].set_xlabel(target)

    sns.boxplot(data=df, x="RiskTier", y=target, ax=axes[1], palette="coolwarm")
    axes[1].set_title(f"{target} by RiskTier")
    fig.tight_layout()
    fig.savefig(out_dir / "02_target_interestrate.png", dpi=150)
    plt.close(fig)


def _plot_numeric(df: pd.DataFrame, col: str, target: str, out_dir: Path) -> None:
    """Histogram + boxplot by RiskTier for a numeric column."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Histogram
    data = df[col].dropna()
    axes[0].hist(data, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{col} — Distribution")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")

    # Boxplot by target
    sns.boxplot(data=df, x=target, y=col, ax=axes[1], palette="coolwarm", showfliers=False)
    axes[1].set_title(f"{col} by {target}")

    fig.tight_layout()
    fig.savefig(out_dir / f"num_{col}.png", dpi=120)
    plt.close(fig)


def _plot_categorical(df: pd.DataFrame, col: str, target: str, out_dir: Path) -> None:
    """Count bar + stacked proportion by RiskTier for a categorical column."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Value counts
    order = df[col].value_counts().index[:15]  # top 15
    sns.countplot(data=df, y=col, order=order, ax=axes[0], color="steelblue")
    axes[0].set_title(f"{col} — Counts")

    # Stacked proportions by RiskTier
    ct = pd.crosstab(df[col], df[target], normalize="index")
    ct = ct.loc[ct.index.isin(order)]
    ct.plot.barh(stacked=True, ax=axes[1], colormap="coolwarm", legend=True)
    axes[1].set_title(f"{col} — RiskTier Proportion")
    axes[1].legend(title="RiskTier", bbox_to_anchor=(1.02, 1), fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / f"cat_{col}.png", dpi=120)
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, num_cols: List[str], out_dir: Path) -> None:
    """Correlation heatmap for numeric columns."""
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr()

    # Show top correlated pairs
    size = min(len(num_cols), 30)
    # Pick columns with highest absolute mean correlation
    mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
    top_cols = mean_abs_corr.head(size).index.tolist()

    fig, ax = plt.subplots(figsize=(max(10, size * 0.5), max(8, size * 0.4)))
    sns.heatmap(
        corr.loc[top_cols, top_cols],
        annot=size <= 20,
        fmt=".2f" if size <= 20 else "",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation Heatmap (Top Features)")
    fig.tight_layout()
    fig.savefig(out_dir / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)
