import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dex_sim.plotting import (
    plot_df_survival_curve,
    plot_comparison_dashboard,
    plot_efficiency_frontier,
    plot_system_dashboard,
    plot_microstructure_explorer,
    plot_symmetry_diagnostics,
    plot_liquidation_heatmap,
    plot_worst_case_autopsy,
    plot_all
)
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
        ecp_slippage_cost=np.random.uniform(0, 1, P)
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
        slippage_cost=np.random.uniform(0, 20, (P, T))
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

def test_plotting_functions_run_without_error(dummy_results):
    multi_res, outdir = dummy_results
    
    # Test comparisons
    plot_df_survival_curve(multi_res, outdir)
    assert os.path.exists(os.path.join(outdir, "solvency_survival_curve.png"))
    
    plot_comparison_dashboard(multi_res, outdir)
    assert os.path.exists(os.path.join(outdir, "comparison_df_boxplot.png"))
    assert os.path.exists(os.path.join(outdir, "comparison_leverage_violin.png"))
    
    plot_efficiency_frontier(multi_res, outdir)
    assert os.path.exists(os.path.join(outdir, "efficiency_frontier.png"))
    
    # Test per-model charts (AES)
    aes_res = multi_res.models["AES"]
    aes_dir = os.path.join(outdir, "AES")
    os.makedirs(aes_dir, exist_ok=True)
    
    # New Dashboards
    plot_system_dashboard(aes_res, aes_dir)
    assert os.path.exists(os.path.join(aes_dir, "AES_system_dashboard.png"))
    
    plot_microstructure_explorer(aes_res, aes_dir)
    # Risk diamond might not generate if no granular data, but function should run
    
    plot_symmetry_diagnostics(aes_res, aes_dir)
    assert os.path.exists(os.path.join(aes_dir, "AES_symmetry_diagnostics.png"))
    
    # Legacy / Specific
    plot_liquidation_heatmap(aes_res, aes_dir)
    assert os.path.exists(os.path.join(aes_dir, "AES_liquidation_heatmap.png"))
    
    plot_worst_case_autopsy(aes_res, aes_dir)
    assert os.path.exists(os.path.join(aes_dir, "AES_autopsy.png"))

def test_plot_all_driver(dummy_results):
    multi_res, outdir = dummy_results
    plot_all(multi_res, outdir)
    
    # Check if subdirectories created
    assert os.path.exists(os.path.join(outdir, "AES"))
    assert os.path.exists(os.path.join(outdir, "FXD"))