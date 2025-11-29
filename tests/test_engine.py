import pytest
import numpy as np
from dex_sim.engine import _simulate_paths_numba, _build_rt_and_mult

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
    
    # w_vol = 0.5, w_le = 0.4, w_sig = 0.1
    # rt = 0.5 * |ret| + 0.4 * amihud + 0.1 * sigma
    
    # Calculate expected RT for first element of first path
    # rt[0,0] = 0.5 * 0.01 + 0.4 * 0 + 0.1 * 0.02 = 0.005 + 0.002 = 0.007
    
    rt, breaker_state, margin_mult = _build_rt_and_mult(
        log_returns, amihud_le, sigmas, breaker_soft, breaker_hard, breaker_multipliers
    )
    
    assert rt.shape == log_returns.shape
    assert breaker_state.shape == log_returns.shape
    assert margin_mult.shape == log_returns.shape
    
    # Check values for the first element
    assert rt[0, 0] == pytest.approx(0.007)
    assert breaker_state[0, 0] == 0
    assert margin_mult[0, 0] == 1.0
    
    # Test a case that should trigger hard breaker
    # Set returns very high
    log_returns_high = np.ones_like(log_returns) * 5.0
    rt_high, state_high, mult_high = _build_rt_and_mult(
        log_returns_high, amihud_le, sigmas, breaker_soft, breaker_hard, breaker_multipliers
    )
    assert np.all(state_high == 2)
    assert np.all(mult_high == 2.0)

def test_simulate_paths_numba_basic():
    # 1 Path, 3 Timesteps
    P, T = 1, 3
    initial_price = 100.0
    notional = 1000.0
    im0 = 100.0 # 10x leverage
    slippage_factor = 0.0
    
    # Scenario: Price goes up 10%, then down 10%
    # t=0: Price=100
    # t=1: ret=0.1 -> Price ~ 110.5. dPnL ~ 105
    # t=2: ret=-0.1
    
    log_returns = np.array([[0.0, 0.1, -0.1]]) # t=0 is usually initial, but loop starts at 1
    # Note: _simulate_paths_numba loop starts at t=1. 
    # log_returns[p, t] is used for step t.
    
    # Pre-calculate pct_returns
    pct_returns = np.exp(log_returns) - 1.0
    dPnL = notional * pct_returns # For checking
    
    margin_mult = np.ones((P, T))
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost
    ) = _simulate_paths_numba(
        log_returns, pct_returns, initial_price, im0, notional, slippage_factor, margin_mult,
        do_partial_liquidation=False
    )
    
    # Check shapes
    assert df_required.shape == (P,)
    assert defaults.shape == (P,)
    assert price_paths.shape == (P, T)
    
    # Check t=0 initialization
    assert price_paths[0, 0] == initial_price
    assert lev_long[0, 0] == notional / im0
    
    # t=1: Price rose. Long wins, Short loses.
    # Price ~ 110.517
    assert price_paths[0, 1] == pytest.approx(initial_price * np.exp(0.1))
    
    # Check dPnL logic for t=1
    # vm = dPnL[0, 1] ~ 1000 * (1.105 - 1) = 105.17
    # Short equity was 100. Loss 105.17.
    # Short pays 100 (max eq). Short equity becomes 0.
    # Remaining 5.17.
    # Since rem > 0, DF required += 5.17 + slippage (0)
    # Default = 1. Dead = True.
    
    vm_t1 = dPnL[0, 1]
    # Note: The new logic checks bankruptcy: eqS < 0.
    # eqS starts at 100. vm = 105.17.
    # eqS becomes 100 - 105.17 = -5.17.
    # eqS < 0 -> Default.
    # df_loss = -eqS = 5.17.
    
    assert df_required[0] == pytest.approx(vm_t1 - 100.0) # Shortfall
    assert defaults[0] == 1
    
    # t=2 should be dead
    assert np.isnan(lev_long[0, 2])
    assert np.isnan(lev_short[0, 2])

def test_simulate_paths_numba_no_default():
    # 1 Path, 2 Timesteps. Zero move -> No Default.
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    im0 = 100.0
    slippage_factor = 0.0
    
    log_returns = np.array([[0.0, 0.0]])
    pct_returns = np.zeros_like(log_returns)
    dPnL = np.zeros_like(log_returns)

    margin_mult = np.ones((P, T))
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost
    ) = _simulate_paths_numba(
        log_returns, pct_returns, initial_price, im0, notional, slippage_factor, margin_mult,
        do_partial_liquidation=False
    )
    
    assert df_required[0] == 0.0
    assert defaults[0] == 0
    
    # Lev should be 1000/100 = 10
    assert lev_long[0, 1] == pytest.approx(10.0)

def test_simulate_paths_numba_margin_call():
    # Test that small move triggers margin call (Default) because MM = IM logic
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    im0 = 100.0
    slippage_factor = 0.0
    
    log_returns = np.array([[0.0, 0.01]]) # 1% move
    pct_returns = np.exp(log_returns) - 1.0
    
    margin_mult = np.ones((P, T))
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost
    ) = _simulate_paths_numba(
        log_returns, pct_returns, initial_price, im0, notional, slippage_factor, margin_mult,
        do_partial_liquidation=False
    )
    
    # 1% move -> PnL ~ 10. EqS ~ 90. Req ~ 100. Shortfall 10.
    # Full Liquidation.
    assert defaults[0] == 1
    assert notional_path[0, 1] == 0.0
    
    # With Partial Liquidation
    (
        df_required_p, defaults_p, _, _, _,
        liquidation_fraction_p, notional_path_p, equity_long_p, equity_short_p, _, _
    ) = _simulate_paths_numba(
        log_returns, pct_returns, initial_price, im0, notional, slippage_factor, margin_mult,
        do_partial_liquidation=True
    )
    
    # Shortfall ~ 10. Notional 1000. k = 10/1000 = 0.01.
    # Should liquidate 1%.
    # Defaults should be 0 (unless k=1).
    # k=0.01 < 1. So default=0?
    # My logic: `if k == 1.0: defaults[p]=1`.
    # So default should be 0.
    
    assert defaults_p[0] == 0
    assert liquidation_fraction_p[0, 1] == pytest.approx(0.01, rel=0.1) # approx
    assert notional_path_p[0, 1] == pytest.approx(1000 * 0.99, rel=0.01)
