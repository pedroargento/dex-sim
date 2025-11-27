from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResults:
    # --- AES regime aggregates ---
    aes_df_required: np.ndarray  # [num_paths] DF $ required if starting at 0
    aes_defaults: np.ndarray  # [num_paths] bool, any VM shortfall/default
    aes_df_exhausted: np.ndarray  # [num_paths] bool (finite DF view)
    aes_haircuts: np.ndarray  # [num_paths] fraction of VM not paid

    # --- FXD regime aggregates ---
    fxd_df_required: np.ndarray
    fxd_defaults: np.ndarray
    fxd_df_exhausted: np.ndarray
    fxd_haircuts: np.ndarray

    # --- Sample path time series (for visualization) ---
    sample_idx: np.ndarray  # [num_sample_paths]
    price_paths: np.ndarray  # [num_sample_paths, horizon]
    leverage_long_aes: np.ndarray  # [num_sample_paths, horizon]
    leverage_short_aes: np.ndarray
    leverage_long_fxd: np.ndarray
    leverage_short_fxd: np.ndarray
    R_aes: np.ndarray  # [num_sample_paths, horizon]
    R_fxd: np.ndarray
    breaker_aes: np.ndarray  # [num_sample_paths, horizon], 0/1/2
    breaker_fxd: np.ndarray

    # --- Metadata ---
    horizon: int
    initial_price: float
