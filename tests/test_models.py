import pytest
from dex_sim.models import AESModel, RiskModel
from dex_sim.models.components import ES_IM, Breaker, FullCloseOut

class DummyModel(RiskModel):
    def initial_margin(self, notional, sigma_daily):
        return 0.0
    def on_price_move(self, *args, **kwargs):
        return 0.0, 0.0, 0.0, False

def test_risk_model_subclass():
    model = DummyModel("dummy")
    assert model.name == "dummy"
    assert model.initial_margin(100, 0.1) == 0.0

def test_aes_model_initial_margin():
    im_comp = ES_IM(conf=0.99, df=6)
    breaker = Breaker()
    liq = FullCloseOut()
    
    model = AESModel("aes_test", im_comp, breaker, liq)
    
    notional = 1000.0
    sigma = 0.02
    
    # Should delegate to im_comp
    expected = im_comp.compute(notional, sigma)
    assert model.initial_margin(notional, sigma) == expected

def test_aes_model_on_price_move():
    im_comp = ES_IM()
    breaker = Breaker()
    liq = FullCloseOut(slippage_factor=0.0)
    
    model = AESModel("aes_test", im_comp, breaker, liq)
    
    # Initial state
    eqL = 100.0
    eqS = 100.0
    notional = 1000.0
    
    # Case 1: Small move up
    dPnL = 10.0
    # Short pays Long 10
    new_eqL, new_eqS, df_loss, default = model.on_price_move(
        eqL, eqS, notional, dPnL, sigma_daily=0.02, amihud=0, R_t=0
    )
    
    assert new_eqL == 110.0
    assert new_eqS == 90.0
    assert df_loss == 0.0
    assert default is False
    
    # Case 2: Large move up causing default
    dPnL = 200.0
    # Short has 100. Can only pay 100.
    # Remainder 100.
    # DF Loss = 100 + slippage(0) = 100.
    
    new_eqL, new_eqS, df_loss, default = model.on_price_move(
        eqL, eqS, notional, dPnL, sigma_daily=0.02, amihud=0, R_t=0
    )
    
    assert new_eqL == 200.0 # Recieves 100 from Short
    assert new_eqS == 0.0
    assert df_loss == 100.0
    assert default is True
