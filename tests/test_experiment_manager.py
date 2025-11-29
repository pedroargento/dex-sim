import pytest
from dex_sim.experiment_manager import build_model
from dex_sim.models import AESModel, FXDModel

def test_build_model_aes():
    cfg = {
        "type": "AES",
        "name": "MyAES",
        "im_conf": 0.95,
        "breaker_soft": 1.1,
        "breaker_hard": 2.2,
        "breaker_mult": [1.0, 1.2, 1.5],
        "slippage": 0.005
    }
    
    model = build_model(cfg)
    assert isinstance(model, AESModel)
    assert model.name == "MyAES"
    assert model.im.conf == 0.95
    assert model.breaker.soft == 1.1
    assert model.liquidation.slippage_factor == 0.005

def test_build_model_fxd():
    cfg = {
        "type": "FXD",
        "leverage": 20.0,
        "slippage": 0.002
    }
    
    model = build_model(cfg)
    assert isinstance(model, FXDModel)
    assert model.name == "FXD_20.0x"
    assert model.im.leverage == 20.0
    assert model.liquidation.slippage_factor == 0.002

def test_build_model_unknown():
    cfg = {"type": "UNKNOWN"}
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(cfg)
