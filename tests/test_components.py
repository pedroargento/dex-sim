import pytest
import numpy as np
from scipy.stats import t
from dex_sim.models.components import ES_IM, FixedLeverageIM, Breaker, FullCloseOut, PartialCloseOut

def test_es_im_compute():
    conf = 0.99
    df = 6
    sigma_daily = 0.02
    notional = 1000.0
    
    es_im = ES_IM(conf=conf, df=df)
    
    t_inv = t.ppf(conf, df)
    expected_es_factor = (t.pdf(t_inv, df) / (1 - conf)) * ((df + t_inv**2) / (df - 1))
    expected_im = sigma_daily * expected_es_factor * notional
    
    computed_im = es_im.compute(notional, sigma_daily)
    assert computed_im == pytest.approx(expected_im, rel=1e-5)

def test_fixed_leverage_im_compute():
    leverage = 10.0
    notional = 1000.0
    sigma_daily = 0.02 # Should be ignored
    
    fl_im = FixedLeverageIM(leverage=leverage)
    expected_im = notional / leverage
    
    assert fl_im.compute(notional, sigma_daily) == expected_im

def test_breaker_margin_multiplier():
    breaker = Breaker(soft=1.0, hard=2.0, multipliers=(1.0, 1.5, 2.0))
    
    assert breaker.margin_multiplier(0.5) == 1.0
    assert breaker.margin_multiplier(1.0) == 1.5
    assert breaker.margin_multiplier(1.5) == 1.5
    assert breaker.margin_multiplier(2.0) == 2.0
    assert breaker.margin_multiplier(2.5) == 2.0

def test_partial_close_out_init():
    pco = PartialCloseOut(slippage_factor=0.005)
    assert pco.slippage_factor == 0.005