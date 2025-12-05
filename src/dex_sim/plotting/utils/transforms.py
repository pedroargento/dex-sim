import numpy as np
import warnings
from typing import Tuple, Optional

def compute_percentiles(
    data: np.ndarray, 
    percentiles: Tuple[int, ...] = (50, 90, 95, 99),
    axis: int = 0
) -> dict:
    """
    Compute percentiles along an axis (default 0 = across paths).
    Returns dict { p: array_of_values }.
    """
    if data is None or data.size == 0:
        return {}
    
    # Check if fully NaN
    if np.all(np.isnan(data)):
        return {}

    res = {}
    # Suppress All-NaN slice warning (common for leverage at t=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for p in percentiles:
            res[p] = np.nanpercentile(data, p, axis=axis)
            
    return res

def reconstruct_im_mm_bands(
    notional_paths: np.ndarray,
    sigma_path: np.ndarray,
    margin_multiplier: np.ndarray,
    im_factor_base: float = 1.0, # Need to know this from model config or heuristic
    im_is_es: bool = True,
    gamma: float = 0.8, # MM factor
    percentiles: Tuple[int, ...] = (50, 90)
) -> dict:
    """
    Reconstruct IM and MM bands.
    IM = notional * sigma * factor * multiplier (if ES)
    IM = notional * factor * multiplier (if Fixed)
    """
    if notional_paths is None:
        return {}

    # We assume sigma_path is (T,) or (P, T). If (T,), broadcast.
    # margin_multiplier is (P, T).
    
    # Align shapes
    P, T = notional_paths.shape
    
    if sigma_path.ndim == 1:
        sigma = np.tile(sigma_path, (P, 1))
    else:
        sigma = sigma_path

    if im_is_es:
        # IM rate per path/step
        im_rate = sigma * im_factor_base * margin_multiplier
    else:
        im_rate = im_factor_base * margin_multiplier

    im_vals = np.abs(notional_paths) * im_rate
    mm_vals = im_vals * gamma

    return {
        "im": compute_percentiles(im_vals, percentiles),
        "mm": compute_percentiles(mm_vals, percentiles)
    }