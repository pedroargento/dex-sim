import pytest
import numpy as np
from dex_sim.engine import _build_rt_and_mult
# Note: importing directly from the numba module to test the kernel
from dex_sim.engine_numba_columnar import _run_simulation_loop_numba_columnar

def test_build_rt_and_mult():
    # P=2, T=3
    log_returns = np.array([
        [0.01, -0.02, 0.05],
        [-0.01, 0.02, -0.05]
    ])
    amihud_le = np.zeros_like(log_returns)
    sigmas = np.ones_like(log_returns) * 0.02
    
    breaker_soft = 1.0
    breaker_hard = 2.0
    breaker_multipliers = (1.0, 1.5, 2.0)
    
    rt, breaker_state, margin_mult = _build_rt_and_mult(
        log_returns, amihud_le, sigmas, breaker_soft, breaker_hard, breaker_multipliers
    )
    
    assert rt.shape == log_returns.shape
    assert breaker_state.shape == log_returns.shape
    assert margin_mult.shape == log_returns.shape
    
    # Check values for the first element
    # rt = 0.5 * |0.01| + 0.4 * 0 + 0.1 * 0.02 = 0.005 + 0.002 = 0.007
    assert rt[0, 0] == pytest.approx(0.007)
    assert breaker_state[0, 0] == 0
    assert margin_mult[0, 0] == 1.0


def run_sim_helper(
    P, T, initial_price, notional_target, sigma_daily,
    log_returns, pct_returns,
    im_factor, im_is_es, slippage=0.001, do_partial=False
):
    # Dummy inputs for other params
    amihud_le = np.zeros((P, T))
    sigmas = np.ones((P, T)) * sigma_daily
    margin_mult_grid = np.ones((P, T))
    breaker_state_grid = np.zeros((P, T), dtype=np.int8)
    
    # Construct Fixed Trader Pool
    # 1 Trader for simplicity in tests
    N = 1
    pool_arrival_tick = np.zeros(N, dtype=np.int64)
    pool_notional = np.full(N, notional_target, dtype=np.float64)
    pool_direction = np.array([1.0]) # Long
    # Equity to give effective leverage.
    # If im_factor is leverage (e.g. 10x), then IM = Notional / 10.
    # Let's set Initial Equity = IM * 2 (safe buffer).
    # But wait, older tests assumed specific setup.
    # Let's assume equity = notional / 5.0 (5x leverage initially)
    pool_equity = np.full(N, notional_target / 5.0, dtype=np.float64)
    
    pool_behavior_id = np.full(N, 2, dtype=np.int64) # Random -> Now needs to be Expander(0) or Reducer(1). Default to Expander
    pool_behavior_id = np.zeros(N, dtype=np.int64) 
    
    expand_rate = 0.01
    reduce_rate = 0.005

    return _run_simulation_loop_numba_columnar(
        log_returns, pct_returns, amihud_le, sigmas, initial_price, sigma_daily,
        notional_target, margin_mult_grid, breaker_state_grid,
        im_factor, im_is_es, slippage, do_partial,
        pool_arrival_tick, pool_notional, pool_direction, pool_equity,
        pool_behavior_id, expand_rate, reduce_rate
    )


def test_simulation_basic():
    # 1 Path, 3 Timesteps
    P, T = 1, 3
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    log_returns = np.array([[0.0, 0.1, -0.1]]) 
    pct_returns = np.exp(log_returns) - 1.0
    
    # Fixed Leverage IM = 10x -> factor = 10.0. But wait, 
    # In engine.py: 
    # if not es: im_factor = leverage
    # In engine_numba: 
    # if not es: if factor>0: base_rate = 1.0/factor
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        _, _, _, _, _
    ) = run_sim_helper(
        P, T, initial_price, notional, sigma_daily,
        log_returns, pct_returns,
        im_factor=0.1, im_is_es=False
    )
    
    assert price_paths[0, 0] == initial_price
    # Initial Lev: Notional(1000) / Equity(200) = 5.0
    # Only Long trader exists in my pool construction above.
    assert lev_long[0, 0] == pytest.approx(5.0)
    
    # t=1: Price +10% (110.5). Long PnL = +105. Eq = 305.
    assert price_paths[0, 1] > 110.0
    assert equity_long[0, 1] > 200.0
    assert defaults[0] == 0
    
    # t=2: Back to ~100. Eq back to ~200.
    assert price_paths[0, 2] == pytest.approx(initial_price)
    assert equity_long[0, 2] == pytest.approx(200.0)


def test_simulation_margin_call():
    # Test that large move triggers margin call
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    # Huge move DOWN: -30%
    # Long trader (constructed in helper) dies.
    log_returns = np.array([[0.0, -0.35]]) # -35%
    pct_returns = np.exp(log_returns) - 1.0
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        _, _, _, _, _
    ) = run_sim_helper(
        P, T, initial_price, notional, sigma_daily,
        log_returns, pct_returns,
        im_factor=0.1, im_is_es=False
    )
    
    # t=0: Eq=200.
    # t=1: Price -30%. PnL = 10 * (70 - 100) = -300.
    # Eq = -100. Default!
    
    # Wait, log(-0.35) -> exp(-0.35) = 0.704. Drop ~30%.
    # PnL ~ 10 * (70.4 - 100) = -296.
    # Eq = 200 - 296 = -96.
    
    assert defaults[0] == 1
    assert df_required[0] > 0.0
    assert equity_long[0, 1] == 0.0


def test_partial_liquidation():
    # Test Partial Liquidation
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    # Move such that Equity < MM but > 0
    # Eq=200. IM=100. MM=80.
    # Need PnL < -120 (to hit 80).
    # Let's drop price by 15%.
    # PnL = 10 * (85 - 100) = -150.
    # Eq = 50.
    # MM is still based on 850 Notional -> 85 IM -> 68 MM.
    # 50 < 68. Liquidation!
    
    log_returns = np.array([[0.0, np.log(0.85)]])
    pct_returns = np.exp(log_returns) - 1.0
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        _, _, _, _, _
    ) = run_sim_helper(
        P, T, initial_price, notional, sigma_daily,
        log_returns, pct_returns,
        im_factor=0.1, im_is_es=False,
        do_partial=True
    )
    
    # Should not default
    assert defaults[0] == 0
    # Should liquidate some fraction
    assert liquidation_fraction[0, 1] > 0.0