from typing import List, Dict

import numpy as np
from numba import njit

from .mc_generator import MCReturnsGenerator
from .data_structures import MultiModelResults, SingleModelResults
from .models import RiskModel, AESModel, FXDModel, FullCloseOut


@njit
def _simulate_paths_numba(
    log_returns: np.ndarray,  # [P, T]
    dPnL: np.ndarray,  # [P, T]
    initial_price: float,
    im0: float,
    notional: float,
    slippage_factor: float,
    margin_mult: np.ndarray,  # [P, T] breaker-based multiplier
) -> tuple[
    np.ndarray,  # df_required [P]
    np.ndarray,  # defaults int [P] (0/1)
    np.ndarray,  # price_paths [P, T]
    np.ndarray,  # lev_long [P, T]
    np.ndarray,  # lev_short [P, T]
]:
    """
    Ultra-fast core: 2-trader symmetric model, numba-compiled.

    Assumptions:
      - One long, one short per path, same IM per side (im0).
      - Defaults occur when loser equity < VM.
      - DF loss = margin_mult[p,t] * (VM shortfall + slippage_factor * notional).
      - After default, path is frozen (price constant, leverage = NaN).
    """
    P, T = log_returns.shape

    df_required = np.zeros(P, dtype=np.float64)
    defaults = np.zeros(P, dtype=np.int64)

    price_paths = np.zeros((P, T), dtype=np.float64)
    lev_long = np.empty((P, T), dtype=np.float64)
    lev_short = np.empty((P, T), dtype=np.float64)

    # initialize first column
    for p in range(P):
        price_paths[p, 0] = initial_price
        lev_long[p, 0] = notional / im0
        lev_short[p, 0] = notional / im0

    nan = np.nan

    for p in range(P):
        eqL = im0
        eqS = im0
        price = initial_price
        defaulted = False

        for t in range(1, T):
            if defaulted:
                # freeze tail and break
                price_paths[p, t] = price
                lev_long[p, t] = nan
                lev_short[p, t] = nan
                # fill remaining timesteps
                for tt in range(t + 1, T):
                    price_paths[p, tt] = price
                    lev_long[p, tt] = nan
                    lev_short[p, tt] = nan
                break

            dlog = log_returns[p, t]
            price *= np.exp(dlog)
            price_paths[p, t] = price

            vm = dPnL[p, t]  # long's PnL (can be + or -)
            if vm == 0.0:
                lev_long[p, t] = notional / eqL if eqL > 0.0 else nan
                lev_short[p, t] = notional / eqS if eqS > 0.0 else nan
                continue

            mult = margin_mult[p, t]

            if vm > 0.0:
                # Long wins, short loses
                vm_abs = vm
                pay = vm_abs if eqS >= vm_abs else eqS
                eqS -= pay
                eqL += pay
                vm_remaining = vm_abs - pay

                if vm_remaining > 0.0:
                    # DF loss scaled by breaker multiplier
                    df_loss = mult * (vm_remaining + slippage_factor * notional)
                    df_required[p] += df_loss
                    defaults[p] = 1
                    defaulted = True
            else:
                # Short wins, long loses
                vm_abs = -vm
                pay = vm_abs if eqL >= vm_abs else eqL
                eqL -= pay
                eqS += pay
                vm_remaining = vm_abs - pay

                if vm_remaining > 0.0:
                    df_loss = mult * (vm_remaining + slippage_factor * notional)
                    df_required[p] += df_loss
                    defaults[p] = 1
                    defaulted = True

            lev_long[p, t] = notional / eqL if eqL > 0.0 else nan
            lev_short[p, t] = notional / eqS if eqS > 0.0 else nan

        # if never defaulted, ensure tail is filled (defensive)
        if not defaulted:
            for t in range(1, T):
                if price_paths[p, t] == 0.0:
                    price_paths[p, t] = price_paths[p, t - 1]
                    lev_long[p, t] = notional / eqL if eqL > 0.0 else nan
                    lev_short[p, t] = notional / eqS if eqS > 0.0 else nan

    return df_required, defaults, price_paths, lev_long, lev_short


def _build_rt_and_margin_mult(
    log_returns: np.ndarray,  # [P, T]
    amihud_le: np.ndarray,  # [P, T]
    sigma_daily: float,
    whale_oi_fraction: float,
    breaker_soft: float,
    breaker_hard: float,
    breaker_multipliers: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build systemic stress index R_t, breaker_state, and margin multiplier:

        R_t ≈ w_vol * |log_return| + w_le * amihud + w_const * sigma_daily * |whale_oi|

    breaker_state codes:
        0 = NORMAL
        1 = SOFT
        2 = HARD

    margin_multiplier:
        NORMAL -> multipliers[0]
        SOFT   -> multipliers[1]
        HARD   -> multipliers[2]
    """

    P, T = log_returns.shape

    # Weights (tunable)
    w_vol = 0.5
    w_le = 0.4
    w_const = 0.1

    const_term = w_const * sigma_daily * abs(whale_oi_fraction)

    # R_t for all paths/times
    rt = w_vol * np.abs(log_returns) + w_le * amihud_le + const_term

    # Breaker state
    breaker_state = np.zeros_like(rt, dtype=np.int8)
    mask_soft = (rt >= breaker_soft) & (rt < breaker_hard)
    mask_hard = rt >= breaker_hard
    breaker_state[mask_soft] = 1
    breaker_state[mask_hard] = 2

    # Margin multipliers
    mult0, mult1, mult2 = breaker_multipliers
    margin_mult = np.empty_like(rt, dtype=np.float64)
    margin_mult[breaker_state == 0] = mult0
    margin_mult[breaker_state == 1] = mult1
    margin_mult[breaker_state == 2] = mult2

    return rt, breaker_state, margin_mult


def run_models_numba(
    models: List[RiskModel],
    num_paths: int = 5_000,
    initial_price: float = 4000.0,
    total_oi: float = 1_000_000_000.0,
    whale_oi_fraction: float = 0.40,
    stress_factor: float = 1.0,
    garch_params_file: str = "garch_params.json",
) -> MultiModelResults:
    """
    Ultra-fast engine:
      - Generates one set of MC paths (log_returns, amihud_le)
      - Precomputes dPnL across all paths
      - Computes R_t and breaker-based margin multipliers for AES-like models
      - Runs a numba-compiled 2-trader simulation for each model

    Notes:
      - AESModel: DF losses are scaled by breaker-based margin multiplier.
      - FXDModel: multiplier = 1, breaker_state=0, rt=0 (no breaker).
      - Hyperliquid-style partial liquidation is NOT handled here; use the
        Python or vectorized engine for HL-type models.
    """

    mc = MCReturnsGenerator(
        garch_params_file=garch_params_file,
        num_paths=num_paths,
        stress_factor=stress_factor,
    )
    log_returns, amihud_le, sigmas = mc.generate_paths(num_paths)
    horizon = mc.horizon

    notional = total_oi * whale_oi_fraction
    sigma_daily = mc.last_sigma_t

    # Precompute dPnL matrix [P, T]
    dPnL = notional * (np.exp(log_returns) - 1.0)

    model_results: Dict[str, SingleModelResults] = {}

    for model in models:
        # Compute initial margin once per model
        im0 = model.initial_margin(notional, sigma_daily)

        # Extract slippage_factor if using FullCloseOut, else 0
        slippage_factor = 0.0
        liquidation = getattr(model, "liquidation", None)
        if isinstance(model, (AESModel, FXDModel)) and isinstance(
            liquidation, FullCloseOut
        ):
            slippage_factor = liquidation.slippage_factor

        # Build R_t, breaker_state, and margin_mult
        if isinstance(model, AESModel):
            breaker = getattr(model, "breaker", None)
            if breaker is not None:
                soft = float(breaker.soft)
                hard = float(breaker.hard)
                mtuple = tuple(float(x) for x in breaker.multipliers)
            else:
                soft, hard, mtuple = 1.0, 2.0, (1.0, 1.5, 2.0)

            rt, breaker_state, margin_mult = _build_rt_and_margin_mult(
                log_returns=log_returns,
                amihud_le=amihud_le,
                sigma_daily=sigma_daily,
                whale_oi_fraction=whale_oi_fraction,
                breaker_soft=soft,
                breaker_hard=hard,
                breaker_multipliers=mtuple,
            )
        else:
            # FXD or other models: no breaker effect
            rt = np.zeros_like(log_returns, dtype=np.float64)
            breaker_state = np.zeros_like(log_returns, dtype=np.int8)
            margin_mult = np.ones_like(log_returns, dtype=np.float64)

        # Call numba core
        df_required, defaults_i, price_paths, lev_long, lev_short = (
            _simulate_paths_numba(
                log_returns=log_returns,
                dPnL=dPnL,
                initial_price=initial_price,
                im0=im0,
                notional=notional,
                slippage_factor=slippage_factor,
                margin_mult=margin_mult,
            )
        )

        defaults_bool = defaults_i.astype(bool)

        model_results[model.name] = SingleModelResults(
            name=model.name,
            df_required=df_required,
            defaults=defaults_bool,
            price_paths=price_paths,
            lev_long=lev_long,
            lev_short=lev_short,
            rt=rt,
            breaker_state=breaker_state,
            margin_multiplier=margin_mult,
            # equity paths, partial_liq, etc., are not tracked in this engine
        )

    return MultiModelResults(
        models=model_results,
        num_paths=num_paths,
        horizon=horizon,
        initial_price=initial_price,
        notional=notional,
        log_returns=log_returns,
        amihud_le=amihud_le,
        sigma_path=sigmas,
        metadata={
            "stress_factor": stress_factor,
            "sigma_daily": sigma_daily,
        },
    )
