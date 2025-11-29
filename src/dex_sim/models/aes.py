# risk_sim/models/aes.py
from __future__ import annotations
from typing import Dict, Any, Tuple

from .base import RiskModel
from .components import InitialMargin, Breaker, FullCloseOut


class AESModel(RiskModel):
    """
    AES-style model:
      - IM based on ES (volatility-sensitive).
      - IM scaled by breaker margin multiplier from R_t.
      - Full close-out on VM shortfall, with DF covering shortfall + slippage.
    """

    def __init__(
        self,
        name: str,
        im: InitialMargin,
        breaker: Breaker,
        liquidation: FullCloseOut,
    ):
        super().__init__(name)
        self.im = im
        self.breaker = breaker
        self.liquidation = liquidation

    def initial_margin(self, notional: float, sigma_daily: float) -> float:
        return self.im.compute(notional, sigma_daily)

    def on_price_move(self, eqL, eqS, notional, dPnL_long, sigma_daily, amihud, R_t):
        vm = abs(dPnL_long)
        df_loss = 0.0
        default = False

        if vm == 0:
            return eqL, eqS, 0.0, False

        if dPnL_long > 0:
            pay = min(eqS, vm)
            eqS -= pay
            eqL += pay
            rem = vm - pay
            if rem > 0:
                df_loss = self.liquidation.df_loss(rem, notional)
                default = True
        else:
            pay = min(eqL, vm)
            eqL -= pay
            eqS += pay
            rem = vm - pay
            if rem > 0:
                df_loss = self.liquidation.df_loss(rem, notional)
                default = True

        return eqL, eqS, df_loss, default
