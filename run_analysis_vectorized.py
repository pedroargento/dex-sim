import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# ============================================================
# ETH-Calibrated Monte Carlo Returns Generator (Vectorized)
# ============================================================


class MCReturnsGenerator:
    """
    Generates Monte Carlo price/sigma/liquidity paths using a GARCH(1,1) model
    calibrated to realistic ETH volatility, vectorized across all paths.

    Uses:
    - Daily GARCH parameters
    - sqrt(dt) scaling to 1-minute returns
    - t-distributed shocks (df=6)
    """

    def __init__(
        self,
        garch_params_file: str = "garch_params.json",
        num_paths: int = 10_000,
        horizon: int = 7_200,  # 5 days * 1440 minutes/day
        df: int = 6,  # ETH-realistic fat tails
        shock_corr: float = -0.7,
        stress_factor: float = 1.0,  # >1.0 to stress shocks
    ):
        with open(garch_params_file, "r") as f:
            params = json.load(f)

        # Daily GARCH parameters
        self.omega = params["omega"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.last_sigma_t = params["last_sigma_t"]  # daily volatility

        self.num_paths = num_paths
        self.horizon = horizon
        self.df = df
        self.stress_factor = stress_factor

        # Price–liquidity correlation
        corr = np.array([[1.0, shock_corr], [shock_corr, 1.0]])
        self.cholesky = np.linalg.cholesky(corr)

        # Daily→minute scaling
        self.TIME_SCALE = np.sqrt(1.0 / 1440.0)  # sqrt(dt) with dt = 1 minute

    def generate_paths(self, num_paths: int | None = None):
        """
        Generate vectorized Monte Carlo paths:

        Returns:
            log_returns: (num_paths, horizon)
            amihud_le:   (num_paths, horizon)
        """
        if num_paths is None:
            num_paths = self.num_paths

        # Heavy-tail t shocks, vectorized across paths & time
        t_indep = np.random.standard_t(self.df, size=(num_paths, self.horizon, 2))

        # Correlate shocks (price & liquidity) and apply stress
        shocks = t_indep @ self.cholesky.T
        shocks *= self.stress_factor

        price_shocks = shocks[:, :, 0]
        liquidity_shocks = shocks[:, :, 1]

        # Storage
        log_returns = np.zeros((num_paths, self.horizon))
        sigmas = np.zeros((num_paths, self.horizon))
        amihud_le = np.zeros((num_paths, self.horizon))

        # Initial sigma (daily)
        sigmas[:, 0] = self.last_sigma_t

        # GARCH recursion (daily), vectorized across paths
        for t_step in range(1, self.horizon):
            sigma_sq = (
                self.omega
                + self.alpha * log_returns[:, t_step - 1] ** 2
                + self.beta * sigmas[:, t_step - 1] ** 2
            )
            sigma = np.sqrt(np.maximum(sigma_sq, 0.0))

            # ETH-realistic daily vol clamp (3%–20%)
            sigma = np.clip(sigma, 0.03, 0.20)
            sigmas[:, t_step] = sigma

            # Convert to 1-min volatility
            sigma_step = sigma * self.TIME_SCALE

            # Minute log-returns (fat-tailed)
            log_returns[:, t_step] = sigma_step * price_shocks[:, t_step]

            # Liquidity proxy (bounded)
            raw = 1.0 / (np.abs(liquidity_shocks[:, t_step]) + 1e-3)
            amihud_le[:, t_step] = np.clip(np.log1p(raw), 0.0, 10.0)

        # Clip extreme returns to keep exp(log_return) finite
        log_returns = np.clip(log_returns, -0.15, 0.15)

        return log_returns, amihud_le


# ============================================================
# Risk Engine – supports vol-based (AES) and fixed leverage (FXD)
# ============================================================


class RiskEngine:
    """
    - Volatility-based IM (for AES): IM = ES(sigma_daily) * notional
    - Fixed-leverage IM (for FXD):   IM = notional / leverage
    """

    def __init__(self, confidence_level: float = 0.99, df: int = 6):
        self.conf = confidence_level
        self.df = df

    def calculate_im_vol_based(
        self, sigma_daily: float, position_notional: float
    ) -> float:
        """
        Vol-based IM via parametric ES of a t-distribution.
        Used for AES model.
        """
        t_inv = t.ppf(self.conf, self.df)
        es_factor = (t.pdf(t_inv, self.df) / (1 - self.conf)) * (
            (self.df + t_inv**2) / (self.df - 1)
        )

        expected_shortfall = sigma_daily * es_factor
        return expected_shortfall * position_notional

    def calculate_im_fixed(self, position_notional: float, leverage: float) -> float:
        """
        Fixed-leverage IM.
        Used for FXD model.
        """
        return position_notional / leverage


# ============================================================
# Systemic Stress Aggregator
# ============================================================


class SystemAggregator:
    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or {"delta": 0.2, "gamma": 0.1, "var": 0.4, "le": 0.3}
        self.thresholds = thresholds or {"soft": 1.5, "hard": 2.5}

    def calculate_rt(self, sys_delta, sys_gamma, sys_var, le_t):
        w = self.weights
        return (
            w["delta"] * np.abs(sys_delta)
            + w["gamma"] * sys_gamma
            + w["var"] * sys_var
            + w["le"] * le_t
        )

    def get_margin_multiplier_array(self, rt):
        soft = self.thresholds["soft"]
        hard = self.thresholds["hard"]
        return np.where(
            rt < soft,
            1.0,
            np.where(rt < hard, 1.5, 2.0),
        )


# ============================================================
# Full Vectorized AES/FXD Simulation + Leverage Tracking
# ============================================================


def run_montecarlo_vectorized(
    num_paths: int = 10_000,
    initial_price: float = 4000.0,
    whale_oi_fraction: float = 0.40,
    total_oi: float = 1_000_000_000.0,  # 1B OI
    initial_default_fund: float = 50_000_000.0,
    fxd_leverage: float = 20.0,
    stress_factor: float = 1.0,
):
    """
    Vectorized Monte Carlo simulation for:
      - AES: volatility-based IM (via ES), dynamic via R_t
      - FXD: fixed-leverage IM (notional / L), fixed maintenance

    Also tracks:
      - default-fund drawdowns (AES & FXD)
      - per-path max effective leverage in both models
        leverage_t = notional / equity_t (while alive)
    """

    # --- Model objects ---
    mc = MCReturnsGenerator(num_paths=num_paths, stress_factor=stress_factor)
    risk = RiskEngine()
    agg = SystemAggregator()

    print(f"Starting vectorized Monte Carlo simulation for {num_paths} paths...")

    # --- Generate returns & liquidity ---
    log_returns, amihud = mc.generate_paths(num_paths)
    horizon = mc.horizon

    # --- Position setup ---
    whale_notional = total_oi * whale_oi_fraction

    # Price path from cumulative log returns
    cum_log = np.cumsum(log_returns, axis=1)  # (paths, horizon)
    price_path = initial_price * np.exp(cum_log)  # (paths, horizon)
    ret_from_entry = price_path / initial_price - 1.0  # (paths, horizon)

    # Short PnL vs entry
    pnl = -whale_notional * ret_from_entry  # (paths, horizon)

    # --- Systemic factors & AES IM ---
    sigma_daily = mc.last_sigma_t
    sys_var = risk.calculate_im_vol_based(sigma_daily, total_oi) / total_oi

    rt = agg.calculate_rt(
        sys_delta=whale_oi_fraction,
        sys_gamma=0.01,
        sys_var=sys_var,
        le_t=amihud,
    )

    margin_multiplier = agg.get_margin_multiplier_array(rt)

    # AES: vol-based IM (dynamic via R_t)
    aes_base_im = risk.calculate_im_vol_based(sigma_daily, whale_notional)
    aes_im_matrix = aes_base_im * margin_multiplier  # (paths, horizon)

    # FXD: fixed-leverage IM (constant)
    fxd_base_im = risk.calculate_im_fixed(whale_notional, fxd_leverage)
    fxd_mm_level = 0.5 * fxd_base_im  # maintenance at 50% IM

    # Trader starts with IM as equity (separate for AES and FXD)
    aes_initial_equity = aes_base_im
    fxd_initial_equity = fxd_base_im

    # --- Default funds & min tracking ---
    aes_df = np.full(num_paths, initial_default_fund)
    fxd_df = np.full(num_paths, initial_default_fund)

    aes_min_df = aes_df.copy()
    fxd_min_df = fxd_df.copy()

    # --- Alive masks ---
    aes_alive = np.ones(num_paths, dtype=bool)
    fxd_alive = np.ones(num_paths, dtype=bool)

    # --- Leverage tracking (max per path) ---
    aes_max_leverage = np.zeros(num_paths)
    fxd_max_leverage = np.zeros(num_paths)

    # --- Time loop — vectorized over paths ---
    for t in range(horizon):
        pnl_t = pnl[:, t]

        # Equity paths (if positions never liquidated)
        aes_equity_t_full = aes_initial_equity + pnl_t
        fxd_equity_t_full = fxd_initial_equity + pnl_t

        # ===== AES =====
        if np.any(aes_alive):
            aes_idx = np.where(aes_alive)[0]
            e = aes_equity_t_full[aes_idx]
            im_t = aes_im_matrix[aes_idx, t]

            # Effective leverage for AES while alive
            # clip equity to avoid division by zero
            safe_e = np.maximum(e, 1e-6)
            lev = whale_notional / safe_e
            aes_max_leverage[aes_idx] = np.maximum(aes_max_leverage[aes_idx], lev)

            # Default condition for AES: equity < IM_t
            aes_default = e < im_t

            # DF covers the margin shortfall IM_t - equity_t
            aes_loss = np.where(aes_default, im_t - e, 0.0)

            aes_df[aes_idx] -= aes_loss
            aes_df[aes_idx] = np.maximum(aes_df[aes_idx], 0.0)
            aes_min_df = np.minimum(aes_min_df, aes_df)

            # Once defaulted, the position is closed for AES
            aes_alive[aes_idx[aes_default]] = False

        # ===== FXD =====
        if np.any(fxd_alive):
            fxd_idx = np.where(fxd_alive)[0]
            e = fxd_equity_t_full[fxd_idx]

            # Effective leverage for FXD while alive
            safe_e = np.maximum(e, 1e-6)
            lev = whale_notional / safe_e
            fxd_max_leverage[fxd_idx] = np.maximum(fxd_max_leverage[fxd_idx], lev)

            # Default condition for FXD: equity < fixed MM level
            fxd_default = e < fxd_mm_level

            # DF covers the margin shortfall MM - equity
            fxd_loss = np.where(fxd_default, fxd_mm_level - e, 0.0)

            fxd_df[fxd_idx] -= fxd_loss
            fxd_df[fxd_idx] = np.maximum(fxd_df[fxd_idx], 0.0)
            fxd_min_df = np.minimum(fxd_min_df, fxd_df)

            # Once defaulted, the position is closed for FXD
            fxd_alive[fxd_idx[fxd_default]] = False

        # Early exit if everyone has defaulted in both models
        if not (aes_alive.any() or fxd_alive.any()):
            break

    # Normalized drawdowns
    aes_drawdowns = (initial_default_fund - aes_min_df) / initial_default_fund
    fxd_drawdowns = (initial_default_fund - fxd_min_df) / initial_default_fund

    # Peak systemic stress (optional)
    rt_peaks = np.max(rt, axis=1)

    return aes_drawdowns, fxd_drawdowns, rt_peaks, aes_max_leverage, fxd_max_leverage


# ============================================================
# Analysis & Plotting (including leverage stats)
# ============================================================


def analyze_results(aes_drawdowns, fxd_drawdowns, aes_max_lev, fxd_max_lev):
    initial_default_fund = 50_000_000

    # --- DF risk metrics ---
    pod_aes = np.mean(aes_drawdowns > 0.5)
    pod_fxd = np.mean(fxd_drawdowns > 0.5)

    var99_aes = np.percentile(aes_drawdowns * initial_default_fund, 99)
    var99_fxd = np.percentile(fxd_drawdowns * initial_default_fund, 99)

    # --- Leverage stats (max over life of position) ---
    def lev_stats(name, lev):
        print(f"{name} max leverage stats:")
        print(f"  mean: {np.mean(lev):.2f}x")
        print(f"  95th pct: {np.percentile(lev, 95):.2f}x")
        print(f"  99th pct: {np.percentile(lev, 99):.2f}x")
        print()

    print("\n--- Analysis Results ---")
    print(f"AES PoD (DF drawdown > 50%): {pod_aes:.2%}")
    print(f"FXD PoD (DF drawdown > 50%): {pod_fxd:.2%}")
    print(f"AES 99% VaR Drawdown: ${var99_aes:,.0f}")
    print(f"FXD 99% VaR Drawdown: ${var99_fxd:,.0f}\n")

    lev_stats("AES", aes_max_lev)
    lev_stats("FXD", fxd_max_lev)

    # --- Drawdown Distribution Plot ---
    plt.figure(figsize=(10, 6))
    plt.hist(aes_drawdowns, bins=50, alpha=0.7, label="AES", color="blue")
    plt.hist(fxd_drawdowns, bins=50, alpha=0.7, label="FXD", color="red")
    plt.legend()
    plt.xlabel("Default Fund Drawdown (fraction of initial)")
    plt.ylabel("Frequency")
    plt.title("Default Fund Drawdown Distribution")
    plt.grid(True)
    plt.savefig("drawdown_distribution.png")
    print("Saved drawdown_distribution.png")

    # --- PoD Bar Chart ---
    plt.figure(figsize=(8, 6))
    labels = ["AES", "FXD"]
    pods = [pod_aes, pod_fxd]
    bars = plt.bar(labels, pods, color=["blue", "red"])
    plt.ylabel("Probability of Default (PoD)")
    plt.title("PoD Comparison (DF drawdown > 50%)")
    plt.grid(axis="y")

    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{y:.2%}",
            ha="center",
            va="bottom",
        )

    plt.savefig("pod_comparison.png")
    print("Saved pod_comparison.png")

    # --- Max leverage histograms (optional) ---
    plt.figure(figsize=(10, 6))
    plt.hist(aes_max_lev, bins=50, alpha=0.7, label="AES max lev", color="blue")
    plt.hist(fxd_max_lev, bins=50, alpha=0.7, label="FXD max lev", color="red")
    plt.legend()
    plt.xlabel("Max Leverage")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Leverage per Path")
    plt.grid(True)
    plt.savefig("max_leverage_distribution.png")
    print("Saved max_leverage_distribution.png")


# ============================================================
# Main Entrypoint
# ============================================================

if __name__ == "__main__":
    aes_draw, fxd_draw, rt_peaks, aes_max_lev, fxd_max_lev = run_montecarlo_vectorized(
        num_paths=10_000,
        fxd_leverage=20.0,  # fixed leverage for FXD model
        stress_factor=1.0,  # >1.0 to stress shocks
    )
    analyze_results(aes_draw, fxd_draw, aes_max_lev, fxd_max_lev)
