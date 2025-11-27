import json
from typing import Optional

import numpy as np
from scipy.stats import t

from data_structures import SimulationResults


# ============================================================
# Monte Carlo returns generator (ETH-calibrated GARCH-ish)
# ============================================================


class MCReturnsGenerator:
    def __init__(
        self,
        garch_params_file: str = "garch_params.json",
        num_paths: int = 10_000,
        horizon: int = 7_200,  # 5 days * 1440 minutes/day
        df: int = 6,
        shock_corr: float = -0.7,
        stress_factor: float = 1.0,
    ):
        with open(garch_params_file, "r") as f:
            params = json.load(f)

        self.omega = params["omega"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.last_sigma_t = params["last_sigma_t"]  # daily vol

        self.num_paths = num_paths
        self.horizon = horizon
        self.df = df
        self.stress_factor = stress_factor

        # Correlation between price & liquidity shocks
        corr = np.array([[1.0, shock_corr], [shock_corr, 1.0]])
        self.cholesky = np.linalg.cholesky(corr)

        # Daily → minute volatility scaling
        self.TIME_SCALE = np.sqrt(1.0 / 1440.0)

    def generate_paths(self, num_paths: Optional[int] = None):
        if num_paths is None:
            num_paths = self.num_paths

        # t-distributed shocks, vectorized
        t_indep = np.random.standard_t(self.df, size=(num_paths, self.horizon, 2))
        shocks = t_indep @ self.cholesky.T
        shocks *= self.stress_factor

        price_shocks = shocks[:, :, 0]
        liquidity_shocks = shocks[:, :, 1]

        log_returns = np.zeros((num_paths, self.horizon))
        sigmas = np.zeros((num_paths, self.horizon))
        amihud_le = np.zeros((num_paths, self.horizon))

        sigmas[:, 0] = self.last_sigma_t

        for t_step in range(1, self.horizon):
            sigma_sq = (
                self.omega
                + self.alpha * log_returns[:, t_step - 1] ** 2
                + self.beta * sigmas[:, t_step - 1] ** 2
            )
            sigma = np.sqrt(np.maximum(sigma_sq, 0.0))
            sigma = np.clip(sigma, 0.03, 0.20)  # keep daily vol realistic
            sigmas[:, t_step] = sigma

            sigma_step = sigma * self.TIME_SCALE
            log_returns[:, t_step] = sigma_step * price_shocks[:, t_step]

            raw = 1.0 / (np.abs(liquidity_shocks[:, t_step]) + 1e-3)
            amihud_le[:, t_step] = np.clip(np.log1p(raw), 0.0, 10.0)

        log_returns = np.clip(log_returns, -0.15, 0.15)
        return log_returns, amihud_le


# ============================================================
# Risk Engine – Vol-based IM (AES) + Fixed-Leverage IM (FXD)
# ============================================================


class RiskEngine:
    def __init__(self, confidence_level: float = 0.99, df: int = 6):
        self.conf = confidence_level
        self.df = df

    def calculate_im_vol_based(
        self, sigma_daily: float, position_notional: float
    ) -> float:
        """
        Volatility-based IM via parametric ES of a t-distribution.
        Used for AES regime.
        """
        t_inv = t.ppf(self.conf, self.df)
        es_factor = (t.pdf(t_inv, self.df) / (1 - self.conf)) * (
            (self.df + t_inv**2) / (self.df - 1)
        )

        expected_shortfall = sigma_daily * es_factor
        return expected_shortfall * position_notional

    def calculate_im_fixed(self, position_notional: float, leverage: float) -> float:
        """
        Fixed-leverage IM. Used for FXD regime.
        """
        return position_notional / leverage


# ============================================================
# 2-Trader AES vs FXD Simulation
# VM-default + slippage close-out
# DF requirement measured as dollar amount (starting at 0)
# PLUS per-path time-series metrics for leverage, R_t, breaker states.
# ============================================================


def run_simulation(
    num_paths: int = 10_000,
    initial_price: float = 4000.0,
    total_oi: float = 1_000_000_000.0,
    whale_oi_fraction: float = 0.40,
    fxd_leverage: float = 20.0,
    initial_default_fund: float = 50_000_000.0,
    stress_factor: float = 1.0,
    slippage_factor: float = 0.001,  # 0.1% of notional extra loss on close-out
    num_sample_paths: int = 5,
    garch_params_file: str = "garch_params.json",
) -> SimulationResults:
    """
    2-trader model (one long, one short) evaluated under:
      - AES regime: vol-based IM (ES * sigma * notional)
      - FXD regime: fixed IM = notional / leverage

    Rules:
      - Same price path, same notional for both regimes.
      - Both traders in each regime start with IM as equity.
      - At each step:
          * Compute incremental PnL (dPnL_long, dPnL_short).
          * Variation margin VM = |dPnL|.
          * Loser pays VM to winner from equity.
          * If loser equity < VM → VM shortfall event:
              - Position is CLOSED in that regime.
              - Total close-out loss = VM_remaining + slippage_loss,
                where slippage_loss = slippage_factor * notional.
              - This total is what DF would need to pay in that regime.
      - We measure DF "drawdown" as:
          df_required[path] = total dollars DF would pay in that path
        as if DF started at 0 and had no cap.
      - IM/MM breaches themselves are NOT treated as defaults; only
        inability to pay VM triggers close-out.

    Additionally, for a subset of sample paths, we track time series of:
      - leverage (long & short) for AES and FXD,
      - systemic risk factor R_t for AES and FXD,
      - breaker state (NORMAL/SOFT/HARD) for AES and FXD,
      - price path.
    """

    # --- Setup generators and risk engine ---
    mc = MCReturnsGenerator(
        garch_params_file=garch_params_file,
        num_paths=num_paths,
        horizon=7_200,
        stress_factor=stress_factor,
    )
    risk = RiskEngine()
    log_returns, amihud_le = mc.generate_paths(num_paths)
    horizon = mc.horizon

    notional = total_oi * whale_oi_fraction

    # IM for AES (vol-based) & FXD (fixed leverage)
    sigma_daily = mc.last_sigma_t
    im_aes = risk.calculate_im_vol_based(sigma_daily, notional)
    im_fxd = risk.calculate_im_fixed(notional, fxd_leverage)

    # Initial equity and DF for both regimes (for haircut logic)
    aes_eq_long = np.full(num_paths, im_aes, dtype=float)
    aes_eq_short = np.full(num_paths, im_aes, dtype=float)
    aes_df = np.full(num_paths, initial_default_fund, dtype=float)
    aes_alive = np.ones(num_paths, dtype=bool)

    fxd_eq_long = np.full(num_paths, im_fxd, dtype=float)
    fxd_eq_short = np.full(num_paths, im_fxd, dtype=float)
    fxd_df = np.full(num_paths, initial_default_fund, dtype=float)
    fxd_alive = np.ones(num_paths, dtype=bool)

    # DF requirement as if starting at 0, unlimited capacity
    aes_df_required = np.zeros(num_paths, dtype=float)
    fxd_df_required = np.zeros(num_paths, dtype=float)

    # Metrics
    aes_default = np.zeros(num_paths, dtype=bool)
    fxd_default = np.zeros(num_paths, dtype=bool)

    aes_vm_theoretical = np.zeros(num_paths, dtype=float)
    aes_vm_paid = np.zeros(num_paths, dtype=float)
    fxd_vm_theoretical = np.zeros(num_paths, dtype=float)
    fxd_vm_paid = np.zeros(num_paths, dtype=float)

    aes_df_exhausted = np.zeros(num_paths, dtype=bool)
    fxd_df_exhausted = np.zeros(num_paths, dtype=bool)

    slippage_loss = slippage_factor * notional

    # ---- Systemic risk / breaker parameters ----
    # Simple R_t: R = w_delta|delta| + w_var*sys_var + w_le*illiquidity
    w_delta = 0.2
    w_var = 0.4
    w_le = 0.4
    sys_delta = whale_oi_fraction

    sys_var_aes = im_aes / total_oi
    sys_var_fxd = im_fxd / total_oi

    breaker_soft = 1.0
    breaker_hard = 2.0

    # ---- Sample paths for time-series tracking ----
    num_sample_paths = min(num_sample_paths, num_paths)
    sample_idx = np.random.choice(num_paths, num_sample_paths, replace=False)

    price_paths = np.zeros((num_sample_paths, horizon))
    price_paths[:, 0] = initial_price

    leverage_long_aes = np.full((num_sample_paths, horizon), np.nan)
    leverage_short_aes = np.full((num_sample_paths, horizon), np.nan)
    leverage_long_fxd = np.full((num_sample_paths, horizon), np.nan)
    leverage_short_fxd = np.full((num_sample_paths, horizon), np.nan)

    R_aes = np.zeros((num_sample_paths, horizon))
    R_fxd = np.zeros((num_sample_paths, horizon))
    breaker_aes = np.zeros(
        (num_sample_paths, horizon), dtype=int
    )  # 0=normal,1=soft,2=hard
    breaker_fxd = np.zeros((num_sample_paths, horizon), dtype=int)

    # initialize leverage at t=0
    for k, idx in enumerate(sample_idx):
        leverage_long_aes[k, 0] = notional / aes_eq_long[idx]
        leverage_short_aes[k, 0] = notional / aes_eq_short[idx]
        leverage_long_fxd[k, 0] = notional / fxd_eq_long[idx]
        leverage_short_fxd[k, 0] = notional / fxd_eq_short[idx]

    # Time loop over steps, vectorized across paths
    for t in range(1, horizon):
        dlog = log_returns[:, t]
        price_change = np.exp(dlog) - 1.0
        dPnL_long = notional * price_change
        vm = np.abs(dPnL_long)

        long_wins = dPnL_long > 0
        short_wins = dPnL_long < 0

        # ============================
        # AES REGIME
        # ============================
        aes_active = aes_alive & (vm > 0.0)
        if np.any(aes_active):
            # Long wins, short loses
            mask = aes_active & long_wins
            if np.any(mask):
                vm_here = vm[mask]
                aes_vm_theoretical[mask] += vm_here

                loser_eq = aes_eq_short[mask]
                pay_from_short = np.minimum(loser_eq, vm_here)
                aes_eq_short[mask] = loser_eq - pay_from_short
                vm_remaining = vm_here - pay_from_short

                default_local = vm_remaining > 0
                if np.any(default_local):
                    full_idx = np.where(mask)[0][default_local]

                    total_loss = vm_remaining[default_local] + slippage_loss
                    aes_df_required[full_idx] += total_loss

                    df_before = aes_df[full_idx]
                    df_pay = np.minimum(df_before, total_loss)
                    aes_df[full_idx] = df_before - df_pay

                    remaining_loss = total_loss - df_pay

                    paid_to_winner = pay_from_short[default_local] + df_pay
                    aes_vm_paid[full_idx] += paid_to_winner
                    aes_eq_long[full_idx] += paid_to_winner

                    exhausted_local = remaining_loss > 0
                    if np.any(exhausted_local):
                        aes_df_exhausted[full_idx[exhausted_local]] = True

                    aes_default[full_idx] = True
                    aes_alive[full_idx] = False

                non_default_local = ~default_local
                if np.any(non_default_local):
                    full_idx_nd = np.where(mask)[0][non_default_local]
                    vm_full = vm_here[non_default_local]
                    aes_vm_paid[full_idx_nd] += vm_full
                    aes_eq_long[full_idx_nd] += vm_full

            # Short wins, long loses
            mask = aes_active & short_wins
            if np.any(mask):
                vm_here = vm[mask]
                aes_vm_theoretical[mask] += vm_here

                loser_eq = aes_eq_long[mask]
                pay_from_long = np.minimum(loser_eq, vm_here)
                aes_eq_long[mask] = loser_eq - pay_from_long
                vm_remaining = vm_here - pay_from_long

                default_local = vm_remaining > 0
                if np.any(default_local):
                    full_idx = np.where(mask)[0][default_local]

                    total_loss = vm_remaining[default_local] + slippage_loss
                    aes_df_required[full_idx] += total_loss

                    df_before = aes_df[full_idx]
                    df_pay = np.minimum(df_before, total_loss)
                    aes_df[full_idx] = df_before - df_pay

                    remaining_loss = total_loss - df_pay

                    paid_to_winner = pay_from_long[default_local] + df_pay
                    aes_vm_paid[full_idx] += paid_to_winner
                    aes_eq_short[full_idx] += paid_to_winner

                    exhausted_local = remaining_loss > 0
                    if np.any(exhausted_local):
                        aes_df_exhausted[full_idx[exhausted_local]] = True

                    aes_default[full_idx] = True
                    aes_alive[full_idx] = False

                non_default_local = ~default_local
                if np.any(non_default_local):
                    full_idx_nd = np.where(mask)[0][non_default_local]
                    vm_full = vm_here[non_default_local]
                    aes_vm_paid[full_idx_nd] += vm_full
                    aes_eq_short[full_idx_nd] += vm_full

        # ============================
        # FXD REGIME
        # ============================
        fxd_active = fxd_alive & (vm > 0.0)
        if np.any(fxd_active):
            # Long wins, short loses
            mask = fxd_active & long_wins
            if np.any(mask):
                vm_here = vm[mask]
                fxd_vm_theoretical[mask] += vm_here

                loser_eq = fxd_eq_short[mask]
                pay_from_short = np.minimum(loser_eq, vm_here)
                fxd_eq_short[mask] = loser_eq - pay_from_short
                vm_remaining = vm_here - pay_from_short

                default_local = vm_remaining > 0
                if np.any(default_local):
                    full_idx = np.where(mask)[0][default_local]

                    total_loss = vm_remaining[default_local] + slippage_loss
                    fxd_df_required[full_idx] += total_loss

                    df_before = fxd_df[full_idx]
                    df_pay = np.minimum(df_before, total_loss)
                    fxd_df[full_idx] = df_before - df_pay

                    remaining_loss = total_loss - df_pay

                    paid_to_winner = pay_from_short[default_local] + df_pay
                    fxd_vm_paid[full_idx] += paid_to_winner
                    fxd_eq_long[full_idx] += paid_to_winner

                    exhausted_local = remaining_loss > 0
                    if np.any(exhausted_local):
                        fxd_df_exhausted[full_idx[exhausted_local]] = True

                    fxd_default[full_idx] = True
                    fxd_alive[full_idx] = False

                non_default_local = ~default_local
                if np.any(non_default_local):
                    full_idx_nd = np.where(mask)[0][non_default_local]
                    vm_full = vm_here[non_default_local]
                    fxd_vm_paid[full_idx_nd] += vm_full
                    fxd_eq_long[full_idx_nd] += vm_full

            # Short wins, long loses
            mask = fxd_active & short_wins
            if np.any(mask):
                vm_here = vm[mask]
                fxd_vm_theoretical[mask] += vm_here

                loser_eq = fxd_eq_long[mask]
                pay_from_long = np.minimum(loser_eq, vm_here)
                fxd_eq_long[mask] = loser_eq - pay_from_long
                vm_remaining = vm_here - pay_from_long

                default_local = vm_remaining > 0
                if np.any(default_local):
                    full_idx = np.where(mask)[0][default_local]

                    total_loss = vm_remaining[default_local] + slippage_loss
                    fxd_df_required[full_idx] += total_loss

                    df_before = fxd_df[full_idx]
                    df_pay = np.minimum(df_before, total_loss)
                    fxd_df[full_idx] = df_before - df_pay

                    remaining_loss = total_loss - df_pay

                    paid_to_winner = pay_from_long[default_local] + df_pay
                    fxd_vm_paid[full_idx] += paid_to_winner
                    fxd_eq_short[full_idx] += paid_to_winner

                    exhausted_local = remaining_loss > 0
                    if np.any(exhausted_local):
                        fxd_df_exhausted[full_idx[exhausted_local]] = True

                    fxd_default[full_idx] = True
                    fxd_alive[full_idx] = False

                non_default_local = ~default_local
                if np.any(non_default_local):
                    full_idx_nd = np.where(mask)[0][non_default_local]
                    vm_full = vm_here[non_default_local]
                    fxd_vm_paid[full_idx_nd] += vm_full
                    fxd_eq_short[full_idx_nd] += vm_full

        # ---- Sample-path metrics (price, leverage, R_t, breaker) ----
        sample_logret = log_returns[sample_idx, t]
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(sample_logret)

        # Compute R_t and breaker state per sample, for both regimes
        for k, idx in enumerate(sample_idx):
            le_t = amihud_le[idx, t]

            # AES R_t
            R_val_aes = w_delta * abs(sys_delta) + w_var * sys_var_aes + w_le * le_t
            R_aes[k, t] = R_val_aes
            if R_val_aes < breaker_soft:
                breaker_aes[k, t] = 0
            elif R_val_aes < breaker_hard:
                breaker_aes[k, t] = 1
            else:
                breaker_aes[k, t] = 2

            # FXD R_t
            R_val_fxd = w_delta * abs(sys_delta) + w_var * sys_var_fxd + w_le * le_t
            R_fxd[k, t] = R_val_fxd
            if R_val_fxd < breaker_soft:
                breaker_fxd[k, t] = 0
            elif R_val_fxd < breaker_hard:
                breaker_fxd[k, t] = 1
            else:
                breaker_fxd[k, t] = 2

            # Leverage (only if still alive with positive equity)
            if aes_alive[idx] and aes_eq_long[idx] > 0:
                leverage_long_aes[k, t] = notional / aes_eq_long[idx]
            else:
                leverage_long_aes[k, t] = np.nan

            if aes_alive[idx] and aes_eq_short[idx] > 0:
                leverage_short_aes[k, t] = notional / aes_eq_short[idx]
            else:
                leverage_short_aes[k, t] = np.nan

            if fxd_alive[idx] and fxd_eq_long[idx] > 0:
                leverage_long_fxd[k, t] = notional / fxd_eq_long[idx]
            else:
                leverage_long_fxd[k, t] = np.nan

            if fxd_alive[idx] and fxd_eq_short[idx] > 0:
                leverage_short_fxd[k, t] = notional / fxd_eq_short[idx]
            else:
                leverage_short_fxd[k, t] = np.nan

        # early exit if both regimes fully dead
        if not (aes_alive.any() or fxd_alive.any()):
            break

    # Haircuts as fraction of theoretical VM (ignoring slippage)
    aes_haircut = np.zeros_like(aes_vm_theoretical)
    fxd_haircut = np.zeros_like(fxd_vm_theoretical)

    mask_aes = aes_vm_theoretical > 0
    aes_haircut[mask_aes] = 1.0 - (aes_vm_paid[mask_aes] / aes_vm_theoretical[mask_aes])

    mask_fxd = fxd_vm_theoretical > 0
    fxd_haircut[mask_fxd] = 1.0 - (fxd_vm_paid[mask_fxd] / fxd_vm_theoretical[mask_fxd])

    return SimulationResults(
        aes_df_required=aes_df_required,
        aes_defaults=aes_default,
        aes_df_exhausted=aes_df_exhausted,
        aes_haircuts=aes_haircut,
        fxd_df_required=fxd_df_required,
        fxd_defaults=fxd_default,
        fxd_df_exhausted=fxd_df_exhausted,
        fxd_haircuts=fxd_haircut,
        sample_idx=sample_idx,
        price_paths=price_paths,
        leverage_long_aes=leverage_long_aes,
        leverage_short_aes=leverage_short_aes,
        leverage_long_fxd=leverage_long_fxd,
        leverage_short_fxd=leverage_short_fxd,
        R_aes=R_aes,
        R_fxd=R_fxd,
        breaker_aes=breaker_aes,
        breaker_fxd=breaker_fxd,
        horizon=horizon,
        initial_price=initial_price,
    )
