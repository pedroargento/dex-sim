# risk_sim/models/components.py
import math
from dataclasses import dataclass

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


@dataclass
class FullCloseOut:
    """
    Full close-out of remaining position, with a simple slippage cost.
    """

    slippage_factor: float = 0.001  # fraction of notional

    def df_loss(self, vm_remaining: float, notional: float) -> float:
        """
        VM shortfall + slippage loss.
        """
        slippage_loss = self.slippage_factor * notional
        return vm_remaining + slippage_loss


@dataclass
class PartialLiquidationHL:
    """
    Hyperliquid-like partial liquidation.

    This is a simplified placeholder:
    - Tries to pay VM from loser equity.
    - If not enough, closes fraction of position to plug gap,
      up to full close-out.
    """

    maintenance_factor: float = 0.005  # e.g. 0.5% MM

    def liquidate(
        self,
        equity_loser: float,
        vm_remaining: float,
        notional: float,
    ) -> tuple[float, float, float]:
        """
        Returns:
            vm_paid_extra  (via position closeout),
            df_loss        (if still undercollateralized),
            new_notional   (after partial/full closeout)
        """
        # Toy implementation: close a fraction proportional to shortfall:
        if vm_remaining <= 0:
            return 0.0, 0.0, notional

        # close up to 20% of notional each liquidation attempt
        close_fraction = min(0.2, vm_remaining / notional)
        close_notional = close_fraction * notional

        vm_paid_extra = close_notional  # assume mark-to-market at par for simplicity
        new_notional = notional - close_notional

        remaining_shortfall = vm_remaining - vm_paid_extra
        df_loss = max(0.0, remaining_shortfall)

        return vm_paid_extra, df_loss, new_notional
