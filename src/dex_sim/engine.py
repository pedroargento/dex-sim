# ============================================================ 
#  engine.py — High-level orchestrator for dex-sim 
# ============================================================ 

import numpy as np

from .mc_generator import MCReturnsGenerator
from .data_structures import MultiModelResults, SingleModelResults
from .models import RiskModel
from .models.components import (
    ES_IM,
    FixedLeverageIM,
    Breaker,
    FullCloseOut,
    PartialCloseOut,
)

# Columnar engine + Rt/Margin builder
from .engine_numba_columnar import (
    _run_simulation_loop_numba_columnar,
)


def _build_rt_and_mult(
    log_returns: np.ndarray,
    amihud_le: np.ndarray,
    sigmas: np.ndarray,
    breaker_soft: float,
    breaker_hard: float,
    breaker_multipliers: tuple,
):
    """
    Build systemic stress index R_t, breaker_state, and margin multiplier.
    Columnar & backend-agnostic.
    """
    P, T = log_returns.shape

    # Tunable weights
    w_vol = 0.5
    w_le = 0.4
    w_sig = 0.1

    vol_term = w_sig * sigmas

    rt = w_vol * np.abs(log_returns) + w_le * amihud_le + vol_term

    # Breaker states
    breaker_state = np.zeros_like(rt, dtype=np.int8)
    breaker_state[(rt >= breaker_soft) & (rt < breaker_hard)] = 1
    breaker_state[rt >= breaker_hard] = 2

    # Multipliers
    mult0, mult1, mult2 = breaker_multipliers
    margin_mult = np.empty_like(rt, dtype=np.float64)
    margin_mult[breaker_state == 0] = mult0
    margin_mult[breaker_state == 1] = mult1
    margin_mult[breaker_state == 2] = mult2

    return rt, breaker_state, margin_mult


# ============================================================ 
#  MAIN ENTRY: run_models() 
# ============================================================ 


def run_models(
    models: list[RiskModel],
    trader_pool: tuple,
    trader_pool_config: dict = None,  # New: pass config to extract rates
    num_paths: int = 5000,
    initial_price: float = 4000.0,
    stress_factor: float = 1.0,
    garch_params_file: str = "garch_params.json",
) -> MultiModelResults:

    # ------------------------------------------------------------ 
    # Step 1 — Generate MC paths (returns, vol, amihud)
    # ------------------------------------------------------------ 
    mc = MCReturnsGenerator(
        garch_params_file=garch_params_file,
        num_paths=num_paths,
        stress_factor=stress_factor,
    )

    log_returns, amihud_le, sigmas = mc.generate_paths(num_paths)
    horizon = mc.horizon
    sigma_daily = mc.last_sigma_t

    pct_returns = np.exp(log_returns) - 1.0

    model_results = {}

    # Unpack trader pool (fixed)
    (
        pool_arrival_tick,
        pool_notional,
        pool_direction,
        pool_equity,
        pool_behavior_id,
    ) = trader_pool

    N = len(pool_arrival_tick)

    # Extract behavior params from pool config or defaults
    behaviors_cfg = {}
    if trader_pool_config and "behaviors" in trader_pool_config:
        behaviors_cfg = trader_pool_config["behaviors"]

    expand_rate = float(behaviors_cfg.get("expand_rate", 0.01))
    reduce_rate = float(behaviors_cfg.get("reduce_rate", 0.005))

    # ------------------------------------------------------------ 
    # Step 2 — Loop through models
    # ------------------------------------------------------------ 
    for model in models:

        print(f"\n=== Running model: {model.name} ===")

        # -------------------------------------------------------- 
        # 2a — Precompute Rt, breaker state, and multipliers
        # -------------------------------------------------------- 
        rt_grid, breaker_state, margin_mult = _build_rt_and_mult(
            log_returns=log_returns,
            amihud_le=amihud_le,
            sigmas=sigmas,
            breaker_soft=model.breaker.soft,
            breaker_hard=model.breaker.hard,
            breaker_multipliers=model.breaker.multipliers,
        )

        # -------------------------------------------------------- 
        # 2b — Prepare parameters for the optimized columnar kernel
        # -------------------------------------------------------- 

        # IM config
        # im_factor & im_is_es allow engine to compute IM without calling Python IM objects
        if model.im.__class__.__name__ == "ES_IM":
            # Compute the true ES factor (dimensionless multiplier)
            # IM = sigma * ES_factor * notional
            # We compute it for sigma=1, notional=1 to get the factor itself.
            im_factor = model.im.compute(notional=1.0, sigma_daily=1.0)
            im_is_es = True
        elif model.im.__class__.__name__ == "FixedLeverageIM":
            im_factor = 1.0 / model.im.leverage
            im_is_es = False
        else:
            # Fallback for unknown IM types (should ideally not happen or raise error)
            im_factor = getattr(model.im, "leverage", 1.0)
            im_is_es = False

        # Liquidation config
        slippage_factor = model.liquidation.slippage_factor
        do_partial = model.liquidation.__class__.__name__ == "PartialCloseOut"

        # -------------------------------------------------------- 
        # 2c — Run the optimized Numba kernel
        # -------------------------------------------------------- 
        (
            df_required,
            defaults_i,
            price_paths,
            lev_long_path,
            lev_short_path,
            liquidation_fraction,
            notional_path,
            equity_long_path,
            equity_short_path,
            df_path,
            slippage_cost,
            intent_normal,
            intent_reduce,
            intent_reject,
            lifetime_i,
            snap_pos_i,
            snap_eq_i,
            snap_im_i,
            snap_mm_i,
            ecp_position_path,
            ecp_slippage_cost,
        ) = _run_simulation_loop_numba_columnar(
            log_returns,
            pct_returns,
            amihud_le,
            sigmas,
            initial_price,
            sigma_daily,
            margin_mult,
            breaker_state,
            # IM config
            im_factor,
            im_is_es,
            # Liq config
            slippage_factor,
            do_partial,
            model.gamma,
            # Trader pool
            pool_arrival_tick,
            pool_notional,
            pool_direction,
            pool_equity,
            pool_behavior_id,
            # Behavior Config
            expand_rate,
            reduce_rate,
        )

        # -------------------------------------------------------- 
        # 2d — Pack into SingleModelResults
        # -------------------------------------------------------- 
        model_results[model.name] = SingleModelResults(
            name=model.name,
            df_required=df_required,
            defaults=defaults_i.astype(bool),
            price_paths=price_paths,
            lev_long=lev_long_path,
            lev_short=lev_short_path,
            rt=rt_grid,
            breaker_state=breaker_state,
            margin_multiplier=margin_mult,
            liquidation_fraction=liquidation_fraction,
            notional_paths=notional_path,
            equity_long=equity_long_path,
            equity_short=equity_short_path,
            df_path=df_path,
            slippage_cost=slippage_cost,
            ecp_position_path=ecp_position_path,
            ecp_slippage_cost=ecp_slippage_cost,
        )

    # ------------------------------------------------------------ 
    # Step 3 — Return MultiModelResults
    # ------------------------------------------------------------ 
    return MultiModelResults(
        models=model_results,
        num_paths=num_paths,
        horizon=horizon,
        initial_price=initial_price,
        notional=0.0, # No global notional anymore
        log_returns=log_returns,
        amihud_le=amihud_le,
        sigma_path=sigmas,
        metadata={"stress_factor": stress_factor, "sigma_daily": sigma_daily},
    )