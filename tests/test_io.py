import os
import shutil
import numpy as np
import pytest
from dex_sim.data_structures import MultiModelResults, SingleModelResults
from dex_sim.results_io import save_results, load_results

def test_save_and_load_full_results(tmp_path):
    # 1. Create a dummy MultiModelResults object with ALL fields populated
    P, T = 2, 5
    
    # arrays
    dummy_arr = np.random.rand(P, T)
    dummy_vec = np.random.rand(P)
    
    # Populate SingleModelResults
    model_res = SingleModelResults(
        name="test_model",
        df_required=dummy_vec.copy(),
        defaults=np.zeros(P, dtype=bool),
        price_paths=dummy_arr.copy(),
        lev_long=dummy_arr.copy(),
        lev_short=dummy_arr.copy(),
        # Fields that were missing before
        slippage_cost=np.ones((P, T)) * 100.0,
        df_path=np.ones((P, T)) * 50.0,
        intent_accepted_normal=np.ones((P, T)) * 10.0,
        intent_accepted_reduce=np.ones((P, T)) * 5.0,
        intent_rejected=np.ones((P, T)) * 1.0,
        liquidation_fraction=np.ones((P, T)) * 0.5,
        ecp_position_path=dummy_arr.copy(),
        ecp_slippage_cost=dummy_vec.copy(),
        # Optional fields that were already there
        equity_long=dummy_arr.copy(),
        equity_short=dummy_arr.copy(),
        notional_paths=dummy_arr.copy(),
    )
    
    results = MultiModelResults(
        models={"test_model": model_res},
        num_paths=P,
        horizon=T,
        initial_price=1000.0,
        notional=1000000.0,
        log_returns=dummy_arr.copy(),
        amihud_le=dummy_arr.copy(),
        sigma_path=dummy_arr.copy(),
    )
    
    # 2. Save
    out_dir = tmp_path / "test_results"
    save_results(results, str(out_dir))
    
    # 3. Load
    loaded_results = load_results(str(out_dir))
    loaded_model = loaded_results.models["test_model"]
    
    # 4. Assertions for the previously missing fields
    np.testing.assert_array_equal(loaded_model.slippage_cost, model_res.slippage_cost)
    np.testing.assert_array_equal(loaded_model.df_path, model_res.df_path)
    np.testing.assert_array_equal(loaded_model.intent_accepted_normal, model_res.intent_accepted_normal)
    np.testing.assert_array_equal(loaded_model.intent_accepted_reduce, model_res.intent_accepted_reduce)
    np.testing.assert_array_equal(loaded_model.intent_rejected, model_res.intent_rejected)
    np.testing.assert_array_equal(loaded_model.liquidation_fraction, model_res.liquidation_fraction)
    
    print("\nI/O Test Passed: All arrays preserved.")
