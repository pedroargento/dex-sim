from typing import List, Dict
import numpy as np
from numba import njit

from .mc_generator import MCReturnsGenerator
from .data_structures import MultiModelResults, SingleModelResults
from .models import RiskModel, FullCloseOut, PartialCloseOut


# ============================================================
#  Numba VM / DF simulation core
# ============================================================


@njit
def _simulate_paths_numba(
    log_returns: np.ndarray,
    pct_returns: np.ndarray,
    initial_price: float,
    im0: float,
    notional: float,
    slippage_factor: float,
    margin_mult: np.ndarray,
    do_partial_liquidation: bool,
):
    """
    Ultra-fast 2-trader symmetric VM/DF simulation.

    - One long, one short per path with equal IM (im0).
    - VM = current_notional * pct_return.
    - Supports Partial Liquidation:
        If equity < required_margin:
            close fraction k = shortfall / notional
            apply slippage
    - Or Full Liquidation (legacy):
        If equity < required_margin (or just bankruptcy in old logic, but now unified):
            close all.
    """
    P, T = log_returns.shape

    # Outputs
    df_required = np.zeros(P, dtype=np.float64)
    defaults = np.zeros(P, dtype=np.int64)

    price_paths = np.zeros((P, T), dtype=np.float64)
    lev_long = np.zeros((P, T), dtype=np.float64)
    lev_short = np.zeros((P, T), dtype=np.float64)

    # New Extended Outputs
    liquidation_fraction = np.zeros((P, T), dtype=np.float64)
    notional_path = np.zeros((P, T), dtype=np.float64)
    equity_long = np.zeros((P, T), dtype=np.float64)
    equity_short = np.zeros((P, T), dtype=np.float64)
    df_path = np.zeros((P, T), dtype=np.float64)
    slippage_cost = np.zeros((P, T), dtype=np.float64)

    nan = np.nan
    im_rate = im0 / notional if notional > 0 else 0.0

    # Initialize first time step
    for p in range(P):
        price_paths[p, 0] = initial_price
        lev_long[p, 0] = notional / im0 if im0 > 0 else nan
        lev_short[p, 0] = notional / im0 if im0 > 0 else nan
        
        notional_path[p, 0] = notional
        equity_long[p, 0] = im0
        equity_short[p, 0] = im0

    for p in range(P):
        eqL = im0
        eqS = im0
        curr_notional = notional
        price = initial_price
        dead = False

        for t in range(1, T):
            if dead:
                price_paths[p, t] = price
                lev_long[p, t] = nan
                lev_short[p, t] = nan
                
                notional_path[p, t] = 0.0
                equity_long[p, t] = eqL
                equity_short[p, t] = eqS
                continue

            # Evolve price
            dlog = log_returns[p, t]
            price *= np.exp(dlog)
            price_paths[p, t] = price

            # VM PnL
            ret = pct_returns[p, t]
            vm = curr_notional * ret

            # Apply VM
            eqL += vm
            eqS -= vm

            # Margin Logic
            # Check solvency/margin against CURRENT notional
            # Margin Mult (Breaker) applies to IM requirement
            req_im = curr_notional * im_rate * margin_mult[p, t]

            # We check both sides. In symmetric book, one usually loses.
            # Check Loser First (optimization, and logic)
            # If vm > 0, Short loses. If vm < 0, Long loses.
            
            sides_to_check = (
                (False, True) if vm > 0 else 
                (True, False) if vm < 0 else 
                (True, True)
            ) # (check_long, check_short)

            # Note: In Numba, tuples/lists are tricky if not constant.
            # We'll just check explicitly.
            
            l_needs_check = vm <= 0
            s_needs_check = vm >= 0
            
            # If partial liquidation happens on one side, notional reduces.
            # Does that save the other side? Yes, req_im reduces.
            
            # Helper for liquidation logic (inlined for Numba)
            # We iterate twice to handle "Check Loser -> then Check Winner (with new notional)"
            # Actually, simply checking both sequentially is fine if we update curr_notional.
            
            # SEQUENCE: Check Long -> Check Short (or vice versa).
            # Better: Check Loser, then Check Winner.
            
            if vm > 0: # Short is loser
                # Check Short
                shortfall = req_im - eqS
                if shortfall > 0:
                    if do_partial_liquidation:
                        k = shortfall / curr_notional
                        if k > 1.0: k = 1.0
                        
                        liq_amt = k * curr_notional
                        cost = liq_amt * slippage_factor
                        
                        eqS -= cost
                        slippage_cost[p, t] += cost
                        df_path[p, t] += cost
                        df_required[p] += cost
                        
                        liquidation_fraction[p, t] = max(liquidation_fraction[p, t], k)
                        curr_notional *= (1.0 - k)
                        
                        if k == 1.0:
                            dead = True
                            defaults[p] = 1
                    else:
                        # Full Closeout
                        dead = True
                        defaults[p] = 1 # Margin breach is default in Full mode?
                        cost = curr_notional * slippage_factor
                        eqS -= cost
                        slippage_cost[p, t] += cost
                        df_path[p, t] += cost
                        df_required[p] += cost
                        curr_notional = 0.0

                # Check Long (if not dead)
                if not dead and curr_notional > 1e-9:
                    req_im = curr_notional * im_rate * margin_mult[p, t]
                    shortfall = req_im - eqL
                    if shortfall > 0:
                         # Logic for Long... (Similar)
                         if do_partial_liquidation:
                            k = shortfall / curr_notional
                            if k > 1.0: k = 1.0
                            liq_amt = k * curr_notional
                            cost = liq_amt * slippage_factor
                            eqL -= cost
                            slippage_cost[p, t] += cost
                            df_path[p, t] += cost
                            df_required[p] += cost
                            liquidation_fraction[p, t] = max(liquidation_fraction[p, t], k)
                            curr_notional *= (1.0 - k)
                            if k == 1.0:
                                dead = True
                                defaults[p] = 1
                         else:
                            dead = True
                            defaults[p] = 1
                            cost = curr_notional * slippage_factor
                            eqL -= cost
                            slippage_cost[p, t] += cost
                            df_path[p, t] += cost
                            df_required[p] += cost
                            curr_notional = 0.0

            else: # vm <= 0, Long is loser (or tie)
                # Check Long first
                shortfall = req_im - eqL
                if shortfall > 0:
                    if do_partial_liquidation:
                        k = shortfall / curr_notional
                        if k > 1.0: k = 1.0
                        liq_amt = k * curr_notional
                        cost = liq_amt * slippage_factor
                        eqL -= cost
                        slippage_cost[p, t] += cost
                        df_path[p, t] += cost
                        df_required[p] += cost
                        liquidation_fraction[p, t] = max(liquidation_fraction[p, t], k)
                        curr_notional *= (1.0 - k)
                        if k == 1.0:
                            dead = True
                            defaults[p] = 1
                    else:
                        dead = True
                        defaults[p] = 1
                        cost = curr_notional * slippage_factor
                        eqL -= cost
                        slippage_cost[p, t] += cost
                        df_path[p, t] += cost
                        df_required[p] += cost
                        curr_notional = 0.0
                
                # Check Short (if not dead)
                if not dead and curr_notional > 1e-9:
                    req_im = curr_notional * im_rate * margin_mult[p, t]
                    shortfall = req_im - eqS
                    if shortfall > 0:
                        if do_partial_liquidation:
                            k = shortfall / curr_notional
                            if k > 1.0: k = 1.0
                            liq_amt = k * curr_notional
                            cost = liq_amt * slippage_factor
                            eqS -= cost
                            slippage_cost[p, t] += cost
                            df_path[p, t] += cost
                            df_required[p] += cost
                            liquidation_fraction[p, t] = max(liquidation_fraction[p, t], k)
                            curr_notional *= (1.0 - k)
                            if k == 1.0:
                                dead = True
                                defaults[p] = 1
                        else:
                            dead = True
                            defaults[p] = 1
                            cost = curr_notional * slippage_factor
                            eqS -= cost
                            slippage_cost[p, t] += cost
                            df_path[p, t] += cost
                            df_required[p] += cost
                            curr_notional = 0.0

            # Terminal Bankruptcy Check (Equity < 0)
            # We check this regardless of 'dead' status to capture the final hole.
            # If we just marked dead, we still need to check if eq < 0.
            
            if eqL < 0:
                dead = True
                defaults[p] = 1
                df_loss = -eqL
                # We assume the hole is covered by DF immediately
                # To avoid double counting if loop continued (which it won't due to dead),
                # we treat it as a one-time loss. 
                # But since we set dead=True, we won't re-enter.
                # However, we must ensure we don't add it multiple times if we were somehow iterating?
                # No, loop continues to next t. 'dead' check at start of loop handles skipping.
                
                # We add the hole to DF usage
                df_path[p, t] += df_loss
                df_required[p] += df_loss
                # Reset equity to 0? Or keep negative? 
                # Keep negative for record.
                
            if eqS < 0:
                dead = True
                defaults[p] = 1
                df_loss = -eqS
                df_path[p, t] += df_loss
                df_required[p] += df_loss

            # Update Paths
            if curr_notional < 1e-9:
                dead = True # effectively dead
            
            notional_path[p, t] = curr_notional
            equity_long[p, t] = eqL
            equity_short[p, t] = eqS
            
            if curr_notional > 0 and eqL > 0:
                lev_long[p, t] = curr_notional / eqL
            else:
                lev_long[p, t] = nan
                
            if curr_notional > 0 and eqS > 0:
                lev_short[p, t] = curr_notional / eqS
            else:
                lev_short[p, t] = nan

    return (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost
    )


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

    # Precompute pct returns for variable notional logic
    pct_returns = np.exp(log_returns) - 1.0

    model_results: Dict[str, SingleModelResults] = {}

    for model in models:
        # Initial margin for this model (per side)
        im0 = model.initial_margin(notional, sigma_daily)

        # Slippage factor (for DF requirement when default occurs)
        slippage = model.liquidation.slippage_factor
        
        # Check if partial liquidation is enabled based on component type
        do_partial = isinstance(model.liquidation, PartialCloseOut)

        # Breaker: get params from model.breaker
        # Even FXD has a Breaker component (defaults to infinite soft/hard)
        soft = model.breaker.soft
        hard = model.breaker.hard
        mults = model.breaker.multipliers

        rt, breaker_state, margin_mult = _build_rt_and_mult(
            log_returns=log_returns,
            amihud_le=amihud_le,
            sigmas=sigmas,
            breaker_soft=soft,
            breaker_hard=hard,
            breaker_multipliers=mults,
        )

        # Run numba kernel
        (
            df_required,
            defaults_i,
            price_paths,
            lev_long,
            lev_short,
            liquidation_fraction,
            notional_path,
            equity_long,
            equity_short,
            df_path,
            slippage_cost,
        ) = _simulate_paths_numba(
            log_returns=log_returns,
            pct_returns=pct_returns,
            initial_price=initial_price,
            im0=im0,
            notional=notional,
            slippage_factor=slippage,
            margin_mult=margin_mult,
            do_partial_liquidation=do_partial,
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
            # New Fields
            liquidation_fraction=liquidation_fraction,
            notional_paths=notional_path,
            equity_long=equity_long,
            equity_short=equity_short,
            df_path=df_path,
            slippage_cost=slippage_cost,
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