# risk_sim/models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class RiskModel(ABC):
    """
    Abstract interface for margining + liquidation behavior
    for a symmetric 2-trader (one long, one short) setup.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def initial_margin(self, notional: float, sigma_daily: float) -> float:
        """
        Initial equity per side given notional and daily vol.
        """
        pass

    @abstractmethod
    def on_price_move(
        self,
        equity_long: float,
        equity_short: float,
        notional: float,
        dPnL_long: float,
        context: Dict[str, Any],
    ) -> Tuple[float, float, float, bool]:
        """
        Called each timestep with the price move dPnL_long for the long side.

        Returns:
            new_equity_long,
            new_equity_short,
            df_required_this_step (>= 0),
            default_event (bool) - did any side default/position close-out?
        """
        pass
