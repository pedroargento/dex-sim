from typing import List, Dict
import numpy as np
from numba import njit

from .mc_generator import MCReturnsGenerator
from .data_structures import MultiModelResults, SingleModelResults
from .models import RiskModel, AESModel, FXDModel, FullCloseOut


# ============================================================
#  Numba VM / DF simulation core
# ============================================================


@njit
def _simulate_paths_numba(
    log_returns: np.ndarray,
    dPnL: np.ndarray,
    initial_price: float,
    im0: float,
    notional: float,
    slippage_factor: float,
    margin_mult: np.ndarray,
):
    """
    Ultra-fast 2-trader symmetric VM/DF simulation.

    - One long, one short per path with equal IM (im0).
    - VM = dPnL (long PnL; short is -VM).
    - If loser can't pay full VM, DF covers shortfall + slippage.
    - After default, path is marked dead (remaining timesteps filled with NaN leverage).
    """
    P, T = log_returns.shape

    df_required = np.zeros(P, dtype=np.float64)
    defaults = np.zeros(P, dtype=np.int64)

    price_paths = np.zeros((P, T), dtype=np.float64)
    lev_long = np.zeros((P, T), dtype=np.float64)
    lev_short = np.zeros((P, T), dtype=np.float64)

    # Initialize first time step
    for p in range(P):
        price_paths[p, 0] = initial_price
        lev_long[p, 0] = notional / im0
        lev_short[p, 0] = notional / im0

    nan = np.nan

    for p in range(P):
        eqL = im0
        eqS = im0
        price = initial_price
        dead = False

        for t in range(1, T):
            if dead:
                price_paths[p, t] = price
                lev_long[p, t] = nan
                lev_short[p, t] = nan
                continue

            # evolve price
            dlog = log_returns[p, t]
            price *= np.exp(dlog)
            price_paths[p, t] = price

            vm = dPnL[p, t]  # long PnL
            mult = margin_mult[p, t]

            if vm > 0.0:
                # Long wins, short loses
                pay = vm if eqS >= vm else eqS
                eqS -= pay
                eqL += pay
                rem = vm - pay
                if rem > 0.0:
                    df_required[p] += mult * (rem + slippage_factor * notional)
                    defaults[p] = 1
                    dead = True
            elif vm < 0.0:
                # Short wins, long loses
                loss = -vm
                pay = loss if eqL >= loss else eqL
                eqL -= pay
                eqS += pay
                rem = loss - pay
                if rem > 0.0:
                    df_required[p] += mult * (rem + slippage_factor * notional)
                    defaults[p] = 1
                    dead = True

            lev_long[p, t] = notional / eqL if eqL > 0.0 else nan
            lev_short[p, t] = notional / eqS if eqS > 0.0 else nan

    return df_required, defaults, price_paths, lev_long, lev_short


# ============================================================
#  Breaker / R_t computation
# ============================================================


def _build_rt_and_mult(
    log_returns: np.ndarray,
    amihud_le: np.ndarray,
    sigmas: np.ndarray,
    breaker_soft: float,
    breaker_hard: float,
    breaker_multipliers: tuple[float, float, float],
):
    """
    Build systemic stress index R_t, breaker_state, and margin multiplier.

    R_t combines:
      - |returns|
      - liquidity (amihud_le)
      - per-path volatility (sigmas)

    breaker_state codes:
      0 = NORMAL
      1 = SOFT
      2 = HARD
    """
    P, T = log_returns.shape

    # Tunable weights
    w_vol = 0.5
    w_le = 0.4
    w_sig = 0.1

    # sigmas is [P, T] already
    vol_term = w_sig * sigmas

    rt = w_vol * np.abs(log_returns) + w_le * amihud_le + vol_term

    breaker_state = np.zeros_like(rt, dtype=np.int8)
    breaker_state[(rt >= breaker_soft) & (rt < breaker_hard)] = 1
    breaker_state[rt >= breaker_hard] = 2

    mult0, mult1, mult2 = breaker_multipliers
    margin_mult = np.empty_like(rt, dtype=np.float64)
    margin_mult[breaker_state == 0] = mult0
    margin_mult[breaker_state == 1] = mult1
    margin_mult[breaker_state == 2] = mult2

    return rt, breaker_state, margin_mult


# ============================================================
#  Main engine
# ============================================================


def run_models_numba(
    models: List[RiskModel],
    num_paths: int = 5_000,
    initial_price: float = 4000.0,
    notional: float = 400_000_000.0,
    stress_factor: float = 1.0,
    garch_params_file: str = "garch_params.json",
) -> MultiModelResults:
    """
    Ultra-fast Monte-Carlo engine.

    - Generates shared MC paths (log_returns, amihud, sigmas).
    - For each model:
        * computes IM
        * builds R_t / breaker / margin multipliers (AES)
        * runs numba core to get DF usage, defaults, leverage, price paths
    """

    mc = MCReturnsGenerator(
        garch_params_file=garch_params_file,
        num_paths=num_paths,
        stress_factor=stress_factor,
    )
    log_returns, amihud_le, sigmas = mc.generate_paths(num_paths)
    horizon = mc.horizon
    sigma_daily = mc.last_sigma_t

    # VM increment for the long side
    dPnL = notional * (np.exp(log_returns) - 1.0)

    model_results: Dict[str, SingleModelResults] = {}

    for model in models:
        # Initial margin for this model (per side)
        im0 = model.initial_margin(notional, sigma_daily)

        # Slippage factor (for DF requirement when default occurs)
        slippage = 0.0
        liquidation = getattr(model, "liquidation", None)
        if isinstance(liquidation, FullCloseOut):
            slippage = liquidation.slippage_factor

        # AES: dynamic breaker
        if isinstance(model, AESModel):
            breaker = getattr(model, "breaker", None)
            if breaker is not None:
                soft = float(breaker.soft)
                hard = float(breaker.hard)
                mults = tuple(float(x) for x in breaker.multipliers)
            else:
                soft, hard, mults = 1.0, 2.0, (1.0, 1.5, 2.0)

            rt, breaker_state, margin_mult = _build_rt_and_mult(
                log_returns=log_returns,
                amihud_le=amihud_le,
                sigmas=sigmas,
                breaker_soft=soft,
                breaker_hard=hard,
                breaker_multipliers=mults,
            )
        else:
            # FXD and others: no breaker effect
            rt = np.zeros_like(log_returns, dtype=np.float64)
            breaker_state = np.zeros_like(log_returns, dtype=np.int8)
            margin_mult = np.ones_like(log_returns, dtype=np.float64)

        # Run numba kernel
        df_required, defaults_i, price_paths, lev_long, lev_short = (
            _simulate_paths_numba(
                log_returns=log_returns,
                dPnL=dPnL,
                initial_price=initial_price,
                im0=im0,
                notional=notional,
                slippage_factor=slippage,
                margin_mult=margin_mult,
            )
        )

        model_results[model.name] = SingleModelResults(
            name=model.name,
            df_required=df_required,
            defaults=defaults_i.astype(bool),
            price_paths=price_paths,
            lev_long=lev_long,
            lev_short=lev_short,
            rt=rt,
            breaker_state=breaker_state,
            margin_multiplier=margin_mult,
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
        metadata={"stress_factor": stress_factor, "sigma_daily": sigma_daily},
    )
