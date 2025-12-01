# risk_sim/models/base.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Union
from .components import InitialMargin, Breaker, LiquidationStrategy, TraderArrival

class RiskModel:
    """
    Component-based Risk Model.
    Composed of:
    - InitialMargin component
    - Breaker component
    - Liquidation component
    - TraderArrival component (optional)
    """

    def __init__(
        self, 
        name: str, 
        im: InitialMargin, 
        breaker: Breaker, 
        liquidation: LiquidationStrategy,
        trader_arrival: TraderArrival = None,
        backend: str = "python"
    ):
        self.name = name
        self.im = im
        self.breaker = breaker
        self.liquidation = liquidation
        self.trader_arrival = trader_arrival or TraderArrival()
        self.backend = backend

    def initial_margin(self, notional: float, sigma_daily: float) -> float:
        """
        Delegate IM calculation to the component.
        """
        return self.im.compute(notional, sigma_daily)