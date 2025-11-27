import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# ============================================================
# ETH-Calibrated Monte Carlo Returns Generator
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

    def generate_paths(self, num_paths: int | None = None):
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
# ============================================================


def run_two_trader_aes_vs_fxd_df_required(
    num_paths: int = 10_000,
    initial_price: float = 4000.0,
    total_oi: float = 1_000_000_000.0,
    whale_oi_fraction: float = 0.40,
    fxd_leverage: float = 20.0,
    initial_default_fund: float = 50_000_000.0,
    stress_factor: float = 1.0,
    slippage_factor: float = 0.001,  # 0.1% of notional extra loss on close-out
):
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
              - If we also track a finite DF (initial_default_fund), and it
                cannot fully cover the loss, the winner gets a haircut.
      - We measure DF "drawdown" as:
          df_required[path] = total dollars DF would pay in that path
        as if DF started at 0 and had no cap.
      - IM/MM breaches themselves are NOT treated as defaults; only
        inability to pay VM triggers close-out.
    """

    mc = MCReturnsGenerator(num_paths=num_paths, stress_factor=stress_factor)
    risk = RiskEngine()
    log_returns, _ = mc.generate_paths(num_paths)
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
                    # global indices for paths where AES default happens here
                    full_idx = np.where(mask)[0][default_local]

                    # Total close-out loss = VM_remaining + slippage loss
                    total_loss = vm_remaining[default_local] + slippage_loss

                    # DF requirement as if starting at 0, uncapped
                    aes_df_required[full_idx] += total_loss

                    # For haircuts, use finite DF
                    df_before = aes_df[full_idx]
                    df_pay = np.minimum(df_before, total_loss)
                    aes_df[full_idx] = df_before - df_pay

                    remaining_loss = total_loss - df_pay

                    # Winner actually receives loser equity + DF_pay
                    paid_to_winner = pay_from_short[default_local] + df_pay
                    aes_vm_paid[full_idx] += paid_to_winner
                    aes_eq_long[full_idx] += paid_to_winner

                    exhausted_local = remaining_loss > 0
                    if np.any(exhausted_local):
                        aes_df_exhausted[full_idx[exhausted_local]] = True

                    aes_default[full_idx] = True
                    aes_alive[full_idx] = False

                # Non-default paths: VM fully paid by short
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

    return {
        "aes": {
            "df_required": aes_df_required,  # DF $ needed if starting from 0
            "default_rate": np.mean(aes_default),
            "df_exhaust_rate": np.mean(aes_df_exhausted),
            "haircut_fraction": aes_haircut,
        },
        "fxd": {
            "df_required": fxd_df_required,
            "default_rate": np.mean(fxd_default),
            "df_exhaust_rate": np.mean(fxd_df_exhausted),
            "haircut_fraction": fxd_haircut,
        },
    }


# ============================================================
# Analysis & Plotting for AES vs FXD
# ============================================================


def analyze_aes_vs_fxd_df_required(results):
    aes = results["aes"]
    fxd = results["fxd"]

    def summarize(name, r):
        df_req = r["df_required"]
        haircut = r["haircut_fraction"]
        default_rate = r["default_rate"]
        df_exhaust_rate = r["df_exhaust_rate"]

        print(f"\n--- {name} regime ---")
        print(f"Path default rate (any VM shortfall): {default_rate:.2%}")
        print(f"Paths with any DF exhaustion (finite DF view): {df_exhaust_rate:.2%}")

        print(f"DF required ($) mean:                 {np.mean(df_req):,.0f}")
        print(f"DF required ($) 95th pct:             {np.percentile(df_req, 95):,.0f}")
        print(f"DF required ($) 99th pct:             {np.percentile(df_req, 99):,.0f}")

        if np.any(haircut > 0):
            nonzero = haircut[haircut > 0]
            print(
                f"Median haircut (paths w/ VM):        {np.percentile(nonzero, 50):.2%}"
            )
            print(
                f"95th pct haircut:                    {np.percentile(nonzero, 95):.2%}"
            )
        else:
            print("No haircuts observed (all VM fully paid given finite DF_init).")

    print("\n=== 2-Trader AES vs FXD (DF requirement, VM-default + slippage) ===")
    summarize("AES", aes)
    summarize("FXD", fxd)

    # Plot DF required distribution
    aes_df_req = aes["df_required"]
    fxd_df_req = fxd["df_required"]

    plt.figure(figsize=(10, 6))
    plt.hist(aes_df_req, bins=50, alpha=0.7, label="AES DF required", color="blue")
    plt.hist(fxd_df_req, bins=50, alpha=0.7, label="FXD DF required", color="red")
    plt.xlabel("DF Required per Path (USD)")
    plt.ylabel("Frequency")
    plt.title("Default Fund Requirement Distribution: AES vs FXD")
    plt.legend()
    plt.grid(True)
    plt.savefig("two_trader_aes_fxd_df_required.png")
    print("Saved two_trader_aes_fxd_df_required.png")

    # Haircut distributions (if any)
    aes_hc = aes["haircut_fraction"]
    fxd_hc = fxd["haircut_fraction"]

    if np.any(aes_hc > 0) or np.any(fxd_hc > 0):
        plt.figure(figsize=(10, 6))
        if np.any(aes_hc > 0):
            plt.hist(
                aes_hc[aes_hc > 0],
                bins=50,
                alpha=0.7,
                label="AES haircuts",
                color="blue",
            )
        if np.any(fxd_hc > 0):
            plt.hist(
                fxd_hc[fxd_hc > 0],
                bins=50,
                alpha=0.7,
                label="FXD haircuts",
                color="red",
            )
        plt.xlabel("Haircut Fraction (1 - paid_VM / theoretical_VM)")
        plt.ylabel("Frequency")
        plt.title("Winner Haircut Distribution: AES vs FXD")
        plt.legend()
        plt.grid(True)
        plt.savefig("two_trader_aes_fxd_haircuts.png")
        print("Saved two_trader_aes_fxd_haircuts.png")


# ============================================================
# Main Entrypoint
# ============================================================

if __name__ == "__main__":
    results = run_two_trader_aes_vs_fxd_df_required(
        num_paths=10_000,
        fxd_leverage=20.0,  # FXD fixed leverage
        stress_factor=1.0,  # raise to 1.5–2.0 for more stress
        slippage_factor=0.001,  # 0.1% notional loss when closing a defaulter
        initial_default_fund=50_000_000.0,  # only for finite-DF haircut view
    )
    analyze_aes_vs_fxd_df_required(results)
