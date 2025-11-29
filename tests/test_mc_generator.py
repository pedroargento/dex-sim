import pytest
import numpy as np
import json
import os
from dex_sim.mc_generator import MCReturnsGenerator

@pytest.fixture
def dummy_garch_params(tmp_path):
    params = {
        "omega": 0.00001,
        "alpha": 0.05,
        "beta": 0.90,
        "last_sigma_t": 0.03
    }
    file_path = tmp_path / "garch_params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    return str(file_path)

def test_mc_generator_initialization(dummy_garch_params):
    gen = MCReturnsGenerator(garch_params_file=dummy_garch_params)
    assert gen.omega == 0.00001
    assert gen.last_sigma_t == 0.03

def test_mc_generator_generate_paths(dummy_garch_params):
    num_paths = 10
    horizon = 100
    gen = MCReturnsGenerator(
        garch_params_file=dummy_garch_params,
        num_paths=num_paths,
        horizon=horizon
    )
    
    log_returns, amihud_le, sigmas = gen.generate_paths()
    
    assert log_returns.shape == (num_paths, horizon)
    assert amihud_le.shape == (num_paths, horizon)
    assert sigmas.shape == (num_paths, horizon)
    
    # Check values are within sanity bounds
    assert np.all(sigmas >= 0.03)
    assert np.all(sigmas <= 0.20)
    
    assert np.all(log_returns >= -0.15)
    assert np.all(log_returns <= 0.15)

def test_mc_generator_stress_factor(dummy_garch_params):
    gen = MCReturnsGenerator(garch_params_file=dummy_garch_params, stress_factor=2.0)
    # Cannot easily assert deterministic output without seeding, 
    # but can check if object property is set
    assert gen.stress_factor == 2.0
