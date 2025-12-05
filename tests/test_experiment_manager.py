import pytest
import math
from dex_sim.experiment_manager import build_model
from dex_sim.models import RiskModel, ES_IM, FixedLeverageIM, FullCloseOut, PartialCloseOut

def test_build_model_aes_style():
    cfg = {
        "name": "MyAES",
        "im": {
            "type": "es",
            "conf": 0.95
        },
        "breaker": {
            "soft": 1.1,
            "hard": 2.2,
            "multipliers": [1.0, 1.2, 1.5]
        },
        "liquidation": {
            "type": "partial",
            "slippage": 0.005
        }
    }
    
    model = build_model(cfg)
    assert isinstance(model, RiskModel)
    assert model.name == "MyAES"
    
    assert isinstance(model.im, ES_IM)
    assert model.im.conf == 0.95
    
    assert model.breaker.soft == 1.1
    assert model.breaker.hard == 2.2
    assert model.breaker.multipliers == (1.0, 1.2, 1.5)
    
    assert isinstance(model.liquidation, PartialCloseOut)
    assert model.liquidation.slippage_factor == 0.005
    assert model.gamma == 0.8 # Default

def test_build_model_with_gamma():
    cfg = {
        "name": "GammaTest",
        "im": {"type": "es"},
        "gamma": 0.6
    }
    model = build_model(cfg)
    assert model.gamma == 0.6

def test_build_model_fxd_style():
    cfg = {
        "name": "FXD_20.0x",
        "im": {
            "type": "fixed_leverage",
            "leverage": 20.0
        },
        "# No breaker specified -> default infinite": {},
        "liquidation": {
            "type": "full",
            "slippage": 0.002
        }
    }
    
    model = build_model(cfg)
    assert isinstance(model, RiskModel)
    assert model.name == "FXD_20.0x"
    
    assert isinstance(model.im, FixedLeverageIM)
    assert model.im.leverage == 20.0
    
    assert model.breaker.soft == float('inf')
    
    assert isinstance(model.liquidation, FullCloseOut)
    assert model.liquidation.slippage_factor == 0.002

def test_build_model_unknown_im():
    cfg = {
        "name": "BadIM",
        "im": {"type": "unknown"},
        "liquidation": {}
    }
    with pytest.raises(ValueError, match="Unknown IM type"):
        build_model(cfg)