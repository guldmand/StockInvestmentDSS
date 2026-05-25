# src/stock_investment_dss/uncertainty/edl_action_dataset_v2.py
"""
EDL Action Dataset v2 for D-IQN-DSS (v3.3).

Reads a combined IQN + HierarchicalDecisionPolicy audit CSV and builds
train/eval datasets for the EDL action uncertainty classifier.

Feature specification
---------------------
Core IQN features (always present, 0 NaN):
    21 quantile features (q10/q25/q50/q75/q90 × 4 actions + cvar10 × 4 + score × 4 ... wait)

    More precisely, the combined audit contains per-action:
        q10, q25, q50, q75, q90, cvar10, score (7 per action × 4 EDL actions = 28)
    Plus: iqn_action_margin (1)
    Total IQN features: 29 features

    Note: We use only the 4 EDL action groups (hold/buy/sell/rebalance).
    iqn_*_changestrategy columns are excluded from features.

Context / portfolio features (always present):
    - selected_size_fraction  (1)
    - cash_weight             (1)

Per-ticker features (NaN for HOLD rows — filled with 0.0):
    - momentum_score, value_score, quality_score, risk_score  (4)
    - price_vs_ma50, price_vs_ma200                           (2)
    Total: 6

Total default feature set: 29 + 2 + 6 = 37 features

Categorical encoding
--------------------
Categorical fields (selected_iqn_action, hierarchical_action_type,
selected_ticker, selected_size) are NOT included in the numeric feature
matrix X. They are preserved as metadata columns only.

This is the transparent approach: feature matrix is purely numeric and
requires no encoding overhead.

Point-in-time safety
--------------------
All features are derived from data available at or before decision_date.
EDL-A labels (hindsight) use future prices for supervision only — not features.

Split
-----
Time-ordered split: first 80% rows → train, last 20% rows → eval.
No random shuffling to preserve temporal order.

References
----------
See docs/EDL_Action_Dataset_v3_3.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_investment_dss.uncertainty.edl_action_labeler_v2 import (
    EDL_LABEL_UNAVAILABLE,
    generate_labels,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

# IQN quantile features: 4 actions × 7 metrics each
_EDL_ACTIONS = ["hold", "buy", "sell", "rebalance"]
_IQN_METRICS = ["q10", "q25", "q50", "q75", "q90", "cvar10", "score"]

IQN_FEATURE_COLS: List[str] = [
    f"iqn_{metric}_{action}" for action in _EDL_ACTIONS for metric in _IQN_METRICS
]  # 28 features
IQN_FEATURE_COLS.append("iqn_action_margin")  # 29 total

# Context / portfolio features (always present)
CONTEXT_FEATURE_COLS: List[str] = [
    "selected_size_fraction",
    "cash_weight",
]

# Per-ticker features (NaN for HOLD rows; filled with 0.0)
TICKER_FEATURE_COLS: List[str] = [
    "momentum_score",
    "value_score",
    "quality_score",
    "risk_score",
    "price_vs_ma50",
    "price_vs_ma200",
]

# Full ordered feature list
ALL_FEATURE_COLS: List[str] = (
    IQN_FEATURE_COLS + CONTEXT_FEATURE_COLS + TICKER_FEATURE_COLS
)

# Metadata columns to preserve (not in feature matrix)
METADATA_COLS: List[str] = [
    "decision_id",
    "date",
    "source_iqn_run_id",
    "selected_iqn_action",
    "hierarchical_action_type",
    "selected_ticker",
    "selected_size",
    "final_recommendation_before_edl",
    "edl_label_mode",
    "edl_label_name",
    "edl_label_id",
]

# Train/eval split fraction
TRAIN_FRACTION = 0.80


# ---------------------------------------------------------------------------
# EDLDatasetV2
# ---------------------------------------------------------------------------


@dataclass
class EDLDatasetV2:
    """
    Full EDL dataset produced by DatasetBuilder.

    Attributes
    ----------
    full_df : pd.DataFrame
        All rows with features + labels + metadata.
    train_df : pd.DataFrame
        First 80% rows (temporal order).
    eval_df : pd.DataFrame
        Last 20% rows (temporal order).
    feature_cols : list[str]
        Ordered list of numeric feature column names.
    label_mode : str
        Label mode used ('hindsight', 'rules', 'iqn_teacher').
    n_unavailable : int
        Rows where label could not be determined.
    warnings : list[str]
        Non-fatal build warnings.
    feature_null_summary : dict
        {col: n_null} for feature columns.
    label_distribution : dict
        {label_name: count} for the full dataset.
    train_label_distribution : dict
    eval_label_distribution : dict
    """

    full_df: pd.DataFrame
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    feature_cols: List[str]
    label_mode: str
    n_unavailable: int
    warnings: List[str]
    feature_null_summary: Dict[str, int]
    label_distribution: Dict[str, int]
    train_label_distribution: Dict[str, int]
    eval_label_distribution: Dict[str, int]

    @property
    def n_total(self) -> int:
        return len(self.full_df)

    @property
    def n_train(self) -> int:
        return len(self.train_df)

    @property
    def n_eval(self) -> int:
        return len(self.eval_df)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------


class EDLDatasetBuilderV2:
    """
    Builds train/eval datasets from a combined IQN + HDP audit CSV.

    Usage
    -----
    builder = EDLDatasetBuilderV2()
    dataset = builder.build(combined_df, label_mode="iqn_teacher")

    Parameters
    ----------
    fill_nan_value : float
        Value to use for NaN-filling per-ticker features (default 0.0).
    train_fraction : float
        Fraction of rows for train split (default 0.80).
    exclude_unavailable : bool
        If True, rows with no label are excluded from train/eval.
        If False, they are preserved with label_id=-1.
    """

    def __init__(
        self,
        fill_nan_value: float = 0.0,
        train_fraction: float = TRAIN_FRACTION,
        exclude_unavailable: bool = True,
    ) -> None:
        self.fill_nan_value = fill_nan_value
        self.train_fraction = train_fraction
        self.exclude_unavailable = exclude_unavailable

    def build(
        self,
        combined_df: pd.DataFrame,
        label_mode: str = "iqn_teacher",
    ) -> EDLDatasetV2:
        """
        Build an EDLDatasetV2 from the combined audit DataFrame.

        Parameters
        ----------
        combined_df : pd.DataFrame
            Full combined audit CSV loaded as DataFrame.
        label_mode : str
            'hindsight' (EDL-A), 'rules' (EDL-B), or 'iqn_teacher' (EDL-C).

        Returns
        -------
        EDLDatasetV2
        """
        logger.info(
            "EDLDatasetBuilderV2: building dataset (label_mode=%s, rows=%d)",
            label_mode,
            len(combined_df),
        )

        # ------------------------------------------------------------------
        # Step 1: Generate labels
        # ------------------------------------------------------------------
        labeled_df, label_warnings = generate_labels(combined_df, label_mode)

        # ------------------------------------------------------------------
        # Step 2: Determine which feature columns are actually present
        # ------------------------------------------------------------------
        feature_cols = self._resolve_feature_cols(labeled_df)
        logger.info(
            "Feature columns: %d total (%d IQN, %d context, %d ticker)",
            len(feature_cols),
            len([c for c in feature_cols if c.startswith("iqn_")]),
            len([c for c in feature_cols if c in CONTEXT_FEATURE_COLS]),
            len([c for c in feature_cols if c in TICKER_FEATURE_COLS]),
        )

        # ------------------------------------------------------------------
        # Step 3: Fill NaN in per-ticker features (0.0 = HOLD/no-ticker state)
        # ------------------------------------------------------------------
        for col in feature_cols:
            if col in TICKER_FEATURE_COLS and labeled_df[col].isna().any():
                n_fill = labeled_df[col].isna().sum()
                logger.debug(
                    "NaN-fill %s: %d rows → %.4f", col, n_fill, self.fill_nan_value
                )
                labeled_df[col] = labeled_df[col].fillna(self.fill_nan_value)

        # ------------------------------------------------------------------
        # Step 4: Null summary
        # ------------------------------------------------------------------
        null_summary = {
            col: int(labeled_df[col].isna().sum())
            for col in feature_cols
            if labeled_df[col].isna().any()
        }
        if null_summary:
            logger.warning(
                "Feature columns with remaining NaN after fill: %s", null_summary
            )
        else:
            logger.info("No remaining NaN in feature columns after fill")

        # ------------------------------------------------------------------
        # Step 5: Assemble full dataset with metadata + features + labels
        # ------------------------------------------------------------------
        meta_cols_present = [c for c in METADATA_COLS if c in labeled_df.columns]
        extra_meta = [
            c
            for c in ["visible_data_cutoff", "eval_step", "pit_split_id"]
            if c in labeled_df.columns
        ]

        out_cols = list(dict.fromkeys(meta_cols_present + extra_meta + feature_cols))
        full_df = labeled_df[out_cols].copy()
        full_df = full_df.reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 6: Exclude unavailable labels if requested
        # ------------------------------------------------------------------
        n_unavailable = int((full_df["edl_label_id"] == -1).sum())
        all_warnings: List[str] = label_warnings[:]

        if self.exclude_unavailable and n_unavailable > 0:
            original_n = len(full_df)
            full_df = full_df[full_df["edl_label_id"] >= 0].reset_index(drop=True)
            w = (
                f"Excluded {n_unavailable} rows with unavailable labels. "
                f"Dataset reduced: {original_n} → {len(full_df)} rows."
            )
            all_warnings.append(w)
            logger.warning(w)

        # ------------------------------------------------------------------
        # Step 7: Temporal split (no shuffle)
        # ------------------------------------------------------------------
        n_total = len(full_df)
        n_train = max(1, int(n_total * self.train_fraction))
        train_df = full_df.iloc[:n_train].copy().reset_index(drop=True)
        eval_df = full_df.iloc[n_train:].copy().reset_index(drop=True)
        logger.info(
            "Split: %d train, %d eval (%.0f%% / %.0f%%)",
            len(train_df),
            len(eval_df),
            100 * len(train_df) / max(1, n_total),
            100 * len(eval_df) / max(1, n_total),
        )

        # ------------------------------------------------------------------
        # Step 8: Label distributions
        # ------------------------------------------------------------------
        label_dist = _label_distribution(full_df)
        train_dist = _label_distribution(train_df)
        eval_dist = _label_distribution(eval_df)

        logger.info("Label distribution (full): %s", label_dist)
        logger.info("Label distribution (train): %s", train_dist)
        logger.info("Label distribution (eval): %s", eval_dist)

        return EDLDatasetV2(
            full_df=full_df,
            train_df=train_df,
            eval_df=eval_df,
            feature_cols=feature_cols,
            label_mode=label_mode,
            n_unavailable=n_unavailable,
            warnings=all_warnings,
            feature_null_summary=null_summary,
            label_distribution=label_dist,
            train_label_distribution=train_dist,
            eval_label_distribution=eval_dist,
        )

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Return ordered list of feature columns present in df.
        Missing columns are noted as warnings but do not block build.
        """
        present: List[str] = []
        missing: List[str] = []
        for col in ALL_FEATURE_COLS:
            if col in df.columns:
                present.append(col)
            else:
                missing.append(col)
        if missing:
            logger.warning(
                "Feature columns absent from combined audit (%d): %s",
                len(missing),
                missing,
            )
        return present


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    if "edl_label_name" not in df.columns:
        return {}
    return df["edl_label_name"].value_counts().to_dict()


def build_summary_json(
    dataset: EDLDatasetV2,
    source_combined_run_id: str,
    output_files: Dict[str, str],
) -> dict:
    """Build a machine-readable summary dict for the dataset build."""
    return {
        "source_combined_run_id": source_combined_run_id,
        "label_mode": dataset.label_mode,
        "n_rows_total": dataset.n_total,
        "n_rows_train": dataset.n_train,
        "n_rows_eval": dataset.n_eval,
        "n_unavailable_labels": dataset.n_unavailable,
        "n_features": dataset.n_features,
        "feature_columns": dataset.feature_cols,
        "feature_null_counts": dataset.feature_null_summary,
        "label_distribution_full": dataset.label_distribution,
        "label_distribution_train": dataset.train_label_distribution,
        "label_distribution_eval": dataset.eval_label_distribution,
        "train_fraction": TRAIN_FRACTION,
        "exclude_unavailable_labels": True,
        "warnings": dataset.warnings,
        "output_files": output_files,
    }


def build_summary_md(summary: dict) -> str:
    """Build human-readable markdown summary."""
    dist_full_rows = "\n".join(
        f"| {lbl} | {cnt} | {100*cnt/max(1,summary['n_rows_total']):.1f}% |"
        for lbl, cnt in summary["label_distribution_full"].items()
    )
    feat_null_rows = (
        "\n".join(
            f"| `{col}` | {cnt} |"
            for col, cnt in summary["feature_null_counts"].items()
        )
        or "| _(none)_ | — |"
    )
    warn_block = ""
    if summary["warnings"]:
        warn_block = "\n## ⚠️ Warnings\n" + "\n".join(
            f"- {w}" for w in summary["warnings"]
        )

    return f"""# EDL Action Dataset v2 (v3.3)

## Source
- Combined IQN + HDP run: `{summary['source_combined_run_id']}`
- Label mode: **{summary['label_mode']}**

## Dataset Size
| Split | Rows |
|-------|------|
| Total | {summary['n_rows_total']} |
| Train | {summary['n_rows_train']} |
| Eval  | {summary['n_rows_eval']} |
| Unavailable labels | {summary['n_unavailable_labels']} |

## Features
- **{summary['n_features']} features** in the numeric feature matrix X
- Categorical fields (selected_iqn_action, hierarchical_action_type,
  selected_ticker, selected_size) preserved as metadata only

### Feature null counts (after NaN-fill)
| Column | Null Count |
|--------|------------|
{feat_null_rows}

## Label Distribution (Full Dataset)
| Label | Count | % |
|-------|-------|---|
{dist_full_rows or '| — | 0 | 0% |'}

## Label Distribution (Train)
{_dist_table(summary['label_distribution_train'])}

## Label Distribution (Eval)
{_dist_table(summary['label_distribution_eval'])}

## Output Files
{chr(10).join(f'- `{k}`: `{v}`' for k, v in summary['output_files'].items())}
{warn_block}

## Notes
- Per-ticker features (momentum_score, price_vs_ma50, etc.) are NaN for HOLD rows
  (no ticker selected). These are filled with 0.0 before training.
- ticker_score and size_score are all-null in current combined audit
  (HDP Stage 2/3 scores not populated for IQN run tickers JPM/XOM/UNH/KO/WMT).
  These columns are therefore excluded from the feature matrix.
- value_score, quality_score, risk_score: all-null (FundamentalFeatureStore
  has placeholder data only for AAPL/MSFT/NVDA/AMZN/GOOGL).
- The reliable full-coverage feature set is the 29 IQN features +
  selected_size_fraction + cash_weight = 31 features.
"""


def _dist_table(dist: dict) -> str:
    if not dist:
        return "| — | 0 |"
    rows = "\n".join(f"| {lbl} | {cnt} |" for lbl, cnt in dist.items())
    return f"| Label | Count |\n|-------|-------|\n{rows}"
