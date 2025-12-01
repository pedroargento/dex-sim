import pytest
import numpy as np
from dex_sim.engine import _run_simulation_loop, _build_rt_and_mult
from dex_sim.models import RiskModel, FixedLeverageIM, Breaker, FullCloseOut, PartialCloseOut, TraderArrival

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

def test_simulation_basic():
    # 1 Path, 3 Timesteps
    P, T = 1, 3
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    # Model Setup
    # IM = 100 (10x leverage)
    model = RiskModel(
        name="TestModel",
        im=FixedLeverageIM(leverage=10.0),
        breaker=Breaker(),
        liquidation=FullCloseOut(),
        trader_arrival=TraderArrival(enabled=False)
    )
    
    # Scenario: Price goes up 10%, then down 10%
    log_returns = np.array([[0.0, 0.1, -0.1]]) 
    pct_returns = np.exp(log_returns) - 1.0
    
    amihud_le = np.zeros((P, T))
    sigmas = np.ones((P, T)) * sigma_daily
    
    margin_mult_grid = np.ones((P, T))
    breaker_state_grid = np.zeros((P, T), dtype=np.int8)
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        trader_lifetimes, trader_snapshots
    ) = _run_simulation_loop(
        model, log_returns, pct_returns, amihud_le, sigmas, initial_price, sigma_daily,
        initial_notional_target=notional,
        margin_mult_grid=margin_mult_grid,
        breaker_state_grid=breaker_state_grid
    )
    
    # Check shapes
    assert df_required.shape == (P,)
    assert defaults.shape == (P,)
    assert price_paths.shape == (P, T)
    
    # Check t=0 initialization
    assert price_paths[0, 0] == initial_price
    # Lev = Notional / Equity. Initial Notional=1000. Initial Equity=200 (Long=100, Short=100? No wait)
    # In _run_simulation_loop:
    # t_long = Trader(equity=0.2*notional, ...) -> 200 equity
    # Wait, the code says: t_long = Trader(equity=initial_notional_target*0.2, ...)
    # This creates 5x leverage initially (1000 notional / 200 equity).
    # Wait, the logic I wrote was:
    # t_long = Trader(equity=initial_notional_target*0.2, position=initial_notional_target/price)
    # Lev = (1000/100 * 100) / 200 = 5.0.
    # Why did I chose 0.2? The prompt didn't specify, I just hardcoded initialization.
    # But FixedLeverageIM(10) implies 10x leverage.
    # The code separates IM requirement from Initial Equity.
    # If I want to test specific leverage, I should check the code behavior.
    # I'll update the expectation: Lev should be 5.0 initially based on my hardcoded 0.2 factor.
    
    assert lev_long[0, 0] == pytest.approx(5.0)
    
    # t=1: Price rose 10% (exp(0.1) ~ 1.105)
    # Long position: 10 units.
    # Price: 100 -> 110.5
    # Long PnL: 10 * 10.5 = 105.
    # Long Equity: 200 + 105 = 305.
    # Short PnL: -105.
    # Short Equity: 200 - 105 = 95.
    
    assert price_paths[0, 1] == pytest.approx(initial_price * np.exp(0.1))
    assert equity_long[0, 1] > 200
    assert equity_short[0, 1] < 200
    
    # Check if default? 
    # Short Equity = 95.
    # IM required? 
    # Notional = 10 * 110.5 = 1105.
    # IM = 1105 / 10 = 110.5. (FixedLeverageIM=10)
    # MM = IM * GAMMA(0.8) = 88.4.
    # Equity(95) > MM(88.4). No Liquidation.
    assert defaults[0] == 0
    
    # t=2: Price drops 10% (exp(-0.1) ~ 0.9048 * 110.5 ~ 100)
    # Price goes back to ~100 (actually exp(0.1)*exp(-0.1) = 1.0).
    # So Price = 100.
    # Long PnL from t=1: 10 * (100 - 110.5) = -105.
    # Long Equity: 305 - 105 = 200.
    # Short Equity: 95 + 105 = 200.
    
    assert price_paths[0, 2] == pytest.approx(initial_price)
    assert equity_long[0, 2] == pytest.approx(200.0)


def test_simulation_margin_call():
    # Test that large move triggers margin call
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    model = RiskModel(
        name="TestModel",
        im=FixedLeverageIM(leverage=10.0),
        breaker=Breaker(),
        liquidation=FullCloseOut(),
        trader_arrival=TraderArrival(enabled=False)
    )
    
    # Huge move: +30%
    # t=0: Eq=200 (Lev 5). MM = 1000/10 * 0.8 = 80.
    # t=1: Price +30%.
    # Short PnL = 10 * (100 - 130) = -300.
    # Short Eq = 200 - 300 = -100.
    # Bankruptcy!
    
    log_returns = np.array([[0.0, 0.3]]) 
    pct_returns = np.exp(log_returns) - 1.0
    
    amihud_le = np.zeros((P, T))
    sigmas = np.ones((P, T)) * sigma_daily
    margin_mult_grid = np.ones((P, T))
    breaker_state_grid = np.zeros((P, T), dtype=np.int8)

    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        trader_lifetimes, trader_snapshots
    ) = _run_simulation_loop(
        model, log_returns, pct_returns, amihud_le, sigmas, initial_price, sigma_daily,
        initial_notional_target=notional,
        margin_mult_grid=margin_mult_grid,
        breaker_state_grid=breaker_state_grid
    )
    
    assert defaults[0] == 1
    assert df_required[0] >= 100.0 # Loss of 100 + slippage
    assert equity_short[0, 1] == 0.0 # Zeroed out after default
    assert notional_path[0, 1] < notional # Reduced due to closeout

def test_partial_liquidation():
    # Test Partial Liquidation
    P, T = 1, 2
    initial_price = 100.0
    notional = 1000.0
    sigma_daily = 0.01
    
    model = RiskModel(
        name="TestModel",
        im=FixedLeverageIM(leverage=10.0), # IM=10% -> MM=8%
        breaker=Breaker(),
        liquidation=PartialCloseOut(),
        trader_arrival=TraderArrival(enabled=False)
    )
    
    # Move such that Equity < MM but Equity > 0
    # Initial Equity = 200 (20%). MM = 80 (8%).
    # Need to lose ~130.
    # PnL = -130. 10 units. Price move +13.
    # Price 100 -> 113.
    
    log_returns = np.array([[0.0, np.log(1.13)]])
    pct_returns = np.exp(log_returns) - 1.0
    
    amihud_le = np.zeros((P, T))
    sigmas = np.ones((P, T)) * sigma_daily
    margin_mult_grid = np.ones((P, T))
    breaker_state_grid = np.zeros((P, T), dtype=np.int8)
    
    (
        df_required, defaults, price_paths, lev_long, lev_short,
        liquidation_fraction, notional_path, equity_long, equity_short, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        trader_lifetimes, trader_snapshots
    ) = _run_simulation_loop(
        model, log_returns, pct_returns, amihud_le, sigmas, initial_price, sigma_daily,
        initial_notional_target=notional,
        margin_mult_grid=margin_mult_grid,
        breaker_state_grid=breaker_state_grid
    )
    
    # Short Equity: 200 - 130 = 70.
    # Notional: 10 * 113 = 1130.
    # IM: 113. MM: 90.4.
    # Shortfall: 90.4 - 70 = 20.4.
    # Fraction k = 20.4 / 1130 ~= 0.018.
    
    assert defaults[0] == 0
    assert liquidation_fraction[0, 1] > 0.0
    assert liquidation_fraction[0, 1] < 1.0
    assert notional_path[0, 1] < 1130.0
    assert notional_path[0, 1] > 0.0