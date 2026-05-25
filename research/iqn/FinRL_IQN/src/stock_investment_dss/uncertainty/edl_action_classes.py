# src/stock_investment_dss/uncertainty/edl_action_classes.py
"""
EDL Action Class Definitions and Configuration (v3.2)

Defines the DSS action class space used by the EDL action uncertainty classifier.
The correct EDL formulation for D-IQN-DSS classifies over high-level DSS actions
(HOLD / BUY / SELL / REBALANCE [/ CHANGE_STRATEGY]), NOT over confidence labels.

This supersedes the v3.1 LOW/MEDIUM/HIGH class space.

References
----------
Sensoy et al. (NeurIPS 2018) "Evidential Deep Learning to Quantify Classification Uncertainty"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Canonical action class lists
# ---------------------------------------------------------------------------

EDL_ACTION_CLASSES_4: List[str] = ["HOLD", "BUY", "SELL", "REBALANCE"]
EDL_ACTION_CLASSES_5: List[str] = [
    "HOLD",
    "BUY",
    "SELL",
    "REBALANCE",
    "CHANGE_STRATEGY",
]

VALID_EDL_VARIANTS = frozenset(["none", "A", "B", "C", "AB", "AC", "BC", "ABC"])
VALID_EDL_LABEL_MODES = frozenset(["hindsight", "rules", "iqn_teacher"])


def get_action_classes(include_change_strategy: bool = False) -> List[str]:
    """Return the ordered list of DSS action class names."""
    return EDL_ACTION_CLASSES_5 if include_change_strategy else EDL_ACTION_CLASSES_4


def get_num_classes(include_change_strategy: bool = False) -> int:
    """Return K (number of classes): 4 or 5."""
    return 5 if include_change_strategy else 4


def action_to_idx(action: str, include_change_strategy: bool = False) -> int:
    """
    Map a DSS action string to its integer class index.

    Parameters
    ----------
    action : str
        One of HOLD / BUY / SELL / REBALANCE [/ CHANGE_STRATEGY].
    include_change_strategy : bool
        Whether CHANGE_STRATEGY is the 5th class.

    Returns
    -------
    int in [0, K-1]

    Raises
    ------
    ValueError if action is unknown.
    """
    classes = get_action_classes(include_change_strategy)
    action_upper = action.upper().strip()
    if action_upper not in classes:
        raise ValueError(f"Unknown action '{action}'. Valid classes: {classes}")
    return classes.index(action_upper)


def idx_to_action(idx: int, include_change_strategy: bool = False) -> str:
    """
    Map an integer class index to the DSS action string.

    Parameters
    ----------
    idx : int
        Class index in [0, K-1].
    include_change_strategy : bool

    Returns
    -------
    str: action name
    """
    classes = get_action_classes(include_change_strategy)
    if not (0 <= idx < len(classes)):
        raise ValueError(
            f"Invalid class index {idx}. Valid range: [0, {len(classes)-1}]"
        )
    return classes[idx]


def is_valid_variant(variant: str) -> bool:
    """Return True if variant string is a recognised EDL variant."""
    return variant.lower() in VALID_EDL_VARIANTS or variant in VALID_EDL_VARIANTS


def parse_variant_members(variant: str) -> List[str]:
    """
    Return the list of sub-model identifiers in an ensemble variant string.

    Examples
    --------
    >>> parse_variant_members("ABC") -> ["A", "B", "C"]
    >>> parse_variant_members("BC")  -> ["B", "C"]
    >>> parse_variant_members("none") -> []
    >>> parse_variant_members("A")   -> ["A"]
    """
    v = variant.upper().strip()
    if v == "NONE":
        return []
    return list(v)  # each character is a member: "ABC" -> ["A", "B", "C"]


# ---------------------------------------------------------------------------
# EDLActionConfig
# ---------------------------------------------------------------------------


@dataclass
class EDLActionConfig:
    """
    Central configuration for the EDL action uncertainty layer.

    All parameters can be set programmatically or loaded from environment
    variables via `EDLActionConfig.from_env()`.
    """

    # System-level toggles
    use_hierarchical_policy: bool = True
    use_edl: bool = False

    # EDL variant: none | A | B | C | AB | AC | BC | ABC
    edl_variant: str = "none"

    # Gate
    gate_enabled: bool = True

    # Class space
    include_change_strategy: bool = False

    # Penalty weights in score function
    uncertainty_lambda: float = 0.5
    disagreement_lambda: float = 0.3

    # Label generation
    label_mode: str = "rules"  # hindsight | rules | iqn_teacher
    horizon_days: int = 20

    # Model checkpoint path (empty = untrained / placeholder)
    model_path: str = ""

    # Derived (set automatically)
    action_classes: List[str] = field(default_factory=list)
    num_classes: int = 4

    def __post_init__(self) -> None:
        self.action_classes = get_action_classes(self.include_change_strategy)
        self.num_classes = get_num_classes(self.include_change_strategy)
        self._validate()

    def _validate(self) -> None:
        if not is_valid_variant(self.edl_variant):
            raise ValueError(
                f"Invalid EDL variant '{self.edl_variant}'. "
                f"Valid options: {sorted(VALID_EDL_VARIANTS)}"
            )
        if self.label_mode not in VALID_EDL_LABEL_MODES:
            raise ValueError(
                f"Invalid label_mode '{self.label_mode}'. "
                f"Valid options: {sorted(VALID_EDL_LABEL_MODES)}"
            )
        if self.uncertainty_lambda < 0:
            raise ValueError("uncertainty_lambda must be >= 0")
        if self.disagreement_lambda < 0:
            raise ValueError("disagreement_lambda must be >= 0")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")

    @property
    def variant_members(self) -> List[str]:
        """Return list of sub-model member IDs in the variant, e.g. ['A','B'] for 'AB'."""
        return parse_variant_members(self.edl_variant)

    @property
    def is_ensemble(self) -> bool:
        return len(self.variant_members) > 1

    @classmethod
    def from_env(cls) -> "EDLActionConfig":
        """
        Construct EDLActionConfig from STOCK_INVESTMENT_DSS_EDL_* environment variables.

        Environment variables
        ---------------------
        STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY  : true/false
        STOCK_INVESTMENT_DSS_USE_EDL                  : true/false
        STOCK_INVESTMENT_DSS_EDL_VARIANT              : none|A|B|C|AB|AC|BC|ABC
        STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED         : true/false
        STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY : true/false
        STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA   : float
        STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_LAMBDA  : float
        STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS         : int
        STOCK_INVESTMENT_DSS_EDL_LABEL_MODE           : hindsight|rules|iqn_teacher
        STOCK_INVESTMENT_DSS_EDL_MODEL_PATH           : path string
        """

        def _bool(key: str, default: bool) -> bool:
            v = os.environ.get(key, "").strip().lower()
            if v in ("1", "true", "yes"):
                return True
            if v in ("0", "false", "no"):
                return False
            return default

        def _float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, str(default)).strip())
            except (ValueError, TypeError):
                return default

        def _int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, str(default)).strip())
            except (ValueError, TypeError):
                return default

        def _str(key: str, default: str) -> str:
            return os.environ.get(key, default).strip() or default

        return cls(
            use_hierarchical_policy=_bool(
                "STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY", True
            ),
            use_edl=_bool("STOCK_INVESTMENT_DSS_USE_EDL", False),
            edl_variant=_str("STOCK_INVESTMENT_DSS_EDL_VARIANT", "none"),
            gate_enabled=_bool("STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED", True),
            include_change_strategy=_bool(
                "STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY", False
            ),
            uncertainty_lambda=_float(
                "STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA", 0.5
            ),
            disagreement_lambda=_float(
                "STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_LAMBDA", 0.3
            ),
            horizon_days=_int("STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS", 20),
            label_mode=_str("STOCK_INVESTMENT_DSS_EDL_LABEL_MODE", "rules"),
            model_path=_str("STOCK_INVESTMENT_DSS_EDL_MODEL_PATH", ""),
        )

    def summary(self) -> dict:
        return {
            "use_hierarchical_policy": self.use_hierarchical_policy,
            "use_edl": self.use_edl,
            "edl_variant": self.edl_variant,
            "gate_enabled": self.gate_enabled,
            "include_change_strategy": self.include_change_strategy,
            "num_classes": self.num_classes,
            "action_classes": self.action_classes,
            "uncertainty_lambda": self.uncertainty_lambda,
            "disagreement_lambda": self.disagreement_lambda,
            "horizon_days": self.horizon_days,
            "label_mode": self.label_mode,
            "model_path": self.model_path,
        }
