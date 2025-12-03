# risk_sim/models/components.py
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t


# ---------- Initial Margin components ----------


class InitialMargin:
    def compute(self, notional: float, sigma_daily: float) -> float:
        raise NotImplementedError


@dataclass
class ES_IM(InitialMargin):
    """
    Parametric ES-based IM: ES(sigma) * notional.
    """

    conf: float = 0.99
    df: int = 6

    def compute(self, notional: float, sigma_daily: float) -> float:
        t_inv = t.ppf(self.conf, self.df)
        es_factor = (t.pdf(t_inv, self.df) / (1 - self.conf)) * (
            (self.df + t_inv**2) / (self.df - 1)
        )
        expected_shortfall = sigma_daily * es_factor
        return expected_shortfall * notional


@dataclass
class FixedLeverageIM(InitialMargin):
    """
    Margin = notional / leverage.
    """

    leverage: float

    def compute(self, notional: float, sigma_daily: float) -> float:
        return notional / self.leverage


# ---------- Breaker / margin multipliers ----------


@dataclass
class Breaker:
    soft: float = 1.0
    hard: float = 2.0
    multipliers: tuple = (1.0, 1.5, 2.0)

    def margin_multiplier(self, R_t: float) -> float:
        if R_t < self.soft:
            return self.multipliers[0]
        elif R_t < self.hard:
            return self.multipliers[1]
        else:
            return self.multipliers[2]


# ---------- Liquidation strategies ----------

class LiquidationStrategy:
    slippage_factor: float = 0.001

@dataclass
class FullCloseOut(LiquidationStrategy):
    """
    Full close-out of remaining position, with a simple slippage cost.
    """
    slippage_factor: float = 0.001

    def df_loss(self, vm_remaining: float, notional: float) -> float:
        """
        VM shortfall + slippage loss.
        """
        slippage_loss = self.slippage_factor * notional
        return vm_remaining + slippage_loss


@dataclass
class PartialCloseOut(LiquidationStrategy):
    """
    Partial liquidation strategy.
    """
    slippage_factor: float = 0.001
    
    # Placeholder for any specific partial logic if needed outside Numba
    # In this architecture, the Numba engine handles the 'how' based on type check or flag


# ---------- Trader Entities ----------

class Trader:
    def __init__(self, equity: float, position: float = 0.0, start_step: int = 0):
        self.equity = equity
        self.position = position  # >0 long, <0 short
        self.im_locked = 0.0
        self.mm_required = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.start_step = start_step

    def reduces_exposure(self, delta_q: float) -> bool:
        # True if trade moves position toward zero, not further away
        # If position is 0, any trade increases exposure (returns False)
        if self.position == 0.0:
            return False
        return (self.position > 0 and delta_q < 0) or (self.position < 0 and delta_q > 0)