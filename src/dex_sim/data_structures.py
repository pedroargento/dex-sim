from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

import numpy as np


# ----------------------------------------------------------------------
# Per-model simulation output
# ----------------------------------------------------------------------


@dataclass
class SingleModelResults:
    """
    Stores full path-level results for one risk model.

    Supports:
      - Python engine
      - Vectorized engine
      - Numba engine

    All arrays are shape:
        [num_paths, horizon]
    unless otherwise noted.
    """

    # Identification
    name: str

    # --- Core Outputs ---
    df_required: np.ndarray  # [P]
    defaults: np.ndarray  # [P] boolean
    price_paths: np.ndarray  # [P, T]
    lev_long: np.ndarray  # [P, T]
    lev_short: np.ndarray  # [P, T]

    # --- Optional advanced stats ---
    # Provided only when engines compute them.
    rt: Optional[np.ndarray] = None  # [P, T] systemic risk index
    breaker_state: Optional[np.ndarray] = (
        None  # [P, T] int code: 0=NORMAL, 1=SOFT, 2=HARD
    )
    margin_multiplier: Optional[np.ndarray] = None  # [P, T]

    # Partial liquidation (HL-like)
    partial_liq_amount: Optional[np.ndarray] = None  # [P, T]
    liquidation_fraction: Optional[np.ndarray] = None  # [P, T] (k)
    notional_paths: Optional[np.ndarray] = None  # [P, T]
    df_path: Optional[np.ndarray] = None  # [P, T]
    slippage_cost: Optional[np.ndarray] = None  # [P, T]

    # Equity paths (optional but very useful)
    equity_long: Optional[np.ndarray] = None  # [P, T]
    equity_short: Optional[np.ndarray] = None  # [P, T]

    # Trade Intent Logs
    intent_accepted_normal: Optional[np.ndarray] = None  # [P, T]
    intent_accepted_reduce: Optional[np.ndarray] = None  # [P, T]
    intent_rejected: Optional[np.ndarray] = None  # [P, T]

    # ECP Data
    ecp_position_path: Optional[np.ndarray] = None  # [P, T]
    ecp_slippage_cost: Optional[np.ndarray] = None  # [P]

    # Granular Data
    trader_lifetimes: Optional[np.ndarray] = None # [Total Traders across all paths]
    trader_snapshots: Optional[List[Dict[str, Any]]] = None # For scatter plots (sample or worst path)

    # General-purpose metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Multi-model results
# ----------------------------------------------------------------------


@dataclass
class MultiModelResults:
    """
    Container for results from multiple risk models under the same Monte Carlo paths.
    """

    models: Dict[str, SingleModelResults]

    # Simulation parameters
    num_paths: int
    horizon: int
    initial_price: float
    notional: float

    # Optionally store the MC inputs (for diagnostics or charting)
    log_returns: Optional[np.ndarray] = None  # [P, T]
    amihud_le: Optional[np.ndarray] = None  # [P, T]
    sigma_path: Optional[np.ndarray] = None  # [P, T] or [T]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, model_name: str) -> SingleModelResults:
        """Convenience: results.get("AES") instead of results.models["AES"]."""
        return self.models[model_name]
