import pytest
from dex_sim.models import RiskModel
from dex_sim.models.components import ES_IM, Breaker, FullCloseOut

def test_risk_model_composition():
    im_comp = ES_IM(conf=0.99, df=6)
    breaker = Breaker()
    liq = FullCloseOut()
    
    model = RiskModel("test_model", im_comp, breaker, liq)
    
    assert model.name == "test_model"
    assert model.im == im_comp
    assert model.breaker == breaker
    assert model.liquidation == liq

def test_risk_model_initial_margin_delegation():
    im_comp = ES_IM(conf=0.99, df=6)
    breaker = Breaker()
    liq = FullCloseOut()
    
    model = RiskModel("test_model", im_comp, breaker, liq)
    
    notional = 1000.0
    sigma = 0.02
    
    # Verify delegation
    expected = im_comp.compute(notional, sigma)
    assert model.initial_margin(notional, sigma) == expected