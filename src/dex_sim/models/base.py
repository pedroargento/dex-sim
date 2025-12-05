# risk_sim/models/base.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Union
from .components import InitialMargin, Breaker, LiquidationStrategy

class RiskModel:
    """
    Component-based Risk Model.
    Composed of:
    - InitialMargin component
    - Breaker component
    - Liquidation component
    """

    def __init__(
        self, 
        name: str, 
        im: InitialMargin, 
        breaker: Breaker, 
        liquidation: LiquidationStrategy,
        backend: str = "python",
        gamma: float = 0.8
    ):
        self.name = name
        self.im = im
        self.breaker = breaker
        self.liquidation = liquidation
        self.backend = backend
        self.gamma = gamma

    def initial_margin(self, notional: float, sigma_daily: float) -> float:
        """
        Delegate IM calculation to the component.
        """
        return self.im.compute(notional, sigma_daily)