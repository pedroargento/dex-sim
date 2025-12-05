import os
import pytest
import numpy as np
from dex_sim.plotting import plot_all, generate_dashboard
from dex_sim.data_structures import MultiModelResults, SingleModelResults

@pytest.fixture
def dummy_results(tmp_path):
    # Create dummy SingleModelResults
    P, T = 100, 50
    
    # Model 1: AES
    df_req1 = np.random.exponential(scale=100, size=P)
    price1 = np.cumprod(1 + np.random.normal(0, 0.01, (P, T)), axis=1) * 100
    rt1 = np.abs(np.random.normal(0, 1, (P, T)))
    bs1 = np.random.choice([0, 1, 2], size=(P, T), p=[0.8, 0.15, 0.05])
    mm1 = np.where(bs1==0, 1.0, np.where(bs1==1, 1.5, 2.0))
    liq1 = np.zeros((P, T))
    liq1[0:10, 10:20] = np.random.uniform(0, 1, (10, 10)) # Some liquidations
    notional1 = np.ones((P, T)) * 1000
    eq1 = np.random.normal(100, 10, (P, T))
    
    res1 = SingleModelResults(
        name="AES",
        df_required=df_req1,
        defaults=np.zeros(P),
        price_paths=price1,
        lev_long=np.zeros((P, T)),
        lev_short=np.zeros((P, T)),
        rt=rt1,
        breaker_state=bs1,
        margin_multiplier=mm1,
        liquidation_fraction=liq1,
        notional_paths=notional1,
        equity_long=eq1,
        equity_short=eq1,
        slippage_cost=np.random.uniform(0, 10, (P, T)),
        ecp_position_path=np.random.uniform(-10, 10, (P, T)),
        ecp_slippage_cost=np.random.uniform(0, 1, P),
        intent_accepted_normal=np.random.uniform(0, 10, (P, T)),
        intent_accepted_reduce=np.random.uniform(0, 10, (P, T)),
        intent_rejected=np.random.uniform(0, 1, (P, T)),
        df_path=np.zeros((P, T)),
    )
    
    # Model 2: FXD
    df_req2 = np.random.exponential(scale=150, size=P) # Higher loss
    price2 = price1.copy()
    
    res2 = SingleModelResults(
        name="FXD",
        df_required=df_req2,
        defaults=np.zeros(P),
        price_paths=price2,
        lev_long=np.zeros((P, T)),
        lev_short=np.zeros((P, T)),
        slippage_cost=np.random.uniform(0, 20, (P, T)),
        intent_accepted_normal=np.random.uniform(0, 10, (P, T)),
        intent_accepted_reduce=np.random.uniform(0, 10, (P, T)),
        intent_rejected=np.random.uniform(0, 1, (P, T)),
        df_path=np.zeros((P, T)),
        ecp_position_path=np.zeros((P, T)),
        ecp_slippage_cost=np.zeros(P)
    )
    
    multi_res = MultiModelResults(
        models={"AES": res1, "FXD": res2},
        num_paths=P,
        horizon=T,
        initial_price=100.0,
        notional=1000.0
    )
    
    outdir = tmp_path / "plots"
    os.makedirs(outdir, exist_ok=True)
    
    return multi_res, str(outdir)

def test_generate_dashboard(dummy_results):
    multi_res, outdir = dummy_results
    
    # Test the new Plotly dashboard generator
    path = generate_dashboard(multi_res, outdir)
    
    assert os.path.exists(path)
    assert path.endswith("dashboard.html")
    assert os.path.getsize(path) > 0

def test_plot_all_legacy_wrapper(dummy_results):
    # plot_all should just call generate_dashboard now
    multi_res, outdir = dummy_results
    plot_all(multi_res, outdir)
    
    # It generates index.html inside outdir? 
    # Looking at dashboard_export.py, default name is dashboard.html
    # But plot_all docs said "Legacy entry point".
    # Let's just check the file exists.
    expected = os.path.join(outdir, "dashboard.html")
    assert os.path.exists(expected)
