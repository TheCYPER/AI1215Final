"""Column type inference for the pipeline."""

from typing import List, Optional, Tuple

import pandas as pd


def infer_column_types(
    df: pd.DataFrame,
    targets: List[str],
    forced_categorical: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Classify columns as numeric or categorical.

    Args:
        df: Input dataframe (features only or full — targets are excluded).
        targets: Target column names to exclude.
        forced_categorical: Numeric-dtype columns that should be treated as categorical.
        drop_columns: Columns to exclude entirely.

    Returns:
        (numeric_columns, categorical_columns)
    """
    forced_categorical = forced_categorical or []
    drop_columns = drop_columns or []

    exclude = set(targets) | set(drop_columns)
    cols = [c for c in df.columns if c not in exclude]

    numeric_cols = []
    categorical_cols = []

    for col in cols:
        if col in forced_categorical:
            categorical_cols.append(col)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols
