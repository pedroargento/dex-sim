import numpy as np
import matplotlib.pyplot as plt
from simulation_engine import MCReturnsGenerator, RiskEngine, SystemAggregator


def run_montecarlo():
    """
    Runs the Monte Carlo simulation for both AES and FXD models.
    """
    # --- Simulation Parameters ---
    num_paths = 1000
    initial_price = 4000
    whale_oi_fraction = 0.40
    total_oi = 1_000_000_000  # Total Open Interest in USD
    whale_position_notional = total_oi * whale_oi_fraction
    initial_default_fund = 50_000_000  # 5% of Total OI

    # --- Model Initializations ---
    mc_generator = MCReturnsGenerator(num_paths=num_paths)
    risk_engine = RiskEngine()
    system_aggregator = SystemAggregator()

    # --- Simulation Loop ---
    aes_drawdowns = []
    fxd_drawdowns = []
    rt_peaks = []

    print(f"Starting Monte Carlo simulation for {num_paths} paths...")
    for i in range(num_paths):
        if (i + 1) % 100 == 0:
            print(f"Simulating path {i + 1}/{num_paths}...")

        log_returns, amihud_le = mc_generator.generate_single_path()

        # --- AES Model Simulation ---
        aes_default_fund = initial_default_fund
        aes_max_drawdown = 0
        current_price = initial_price

        # --- FXD Model Simulation ---
        fxd_default_fund = initial_default_fund
        fxd_max_drawdown = 0
        fxd_mm_level = whale_position_notional * 0.05  # 5% MM for 20x leverage

        rt_path = []

        # Cache IM values to avoid massive repeated computation
        sigma_t = mc_generator.last_sigma_t
        sys_var = risk_engine.calculate_im(sigma_t, total_oi) / total_oi
        base_im_whale = risk_engine.calculate_im(sigma_t, whale_position_notional)

        for t_step in range(mc_generator.horizon):
            price_change = np.exp(log_returns[t_step]) - 1
            current_price *= 1 + price_change

            if not np.isfinite(current_price):
                aes_max_drawdown = initial_default_fund
                fxd_max_drawdown = initial_default_fund
                break

            # --- AES Logic ---
            pnl = -whale_position_notional * price_change  # Whale is short

            # Simplified systemic metrics for R_t calculation
            sys_delta = whale_oi_fraction
            sys_gamma = 0.01  # Placeholder

            rt = system_aggregator.calculate_rt(
                sys_delta, sys_gamma, sys_var, amihud_le[t_step]
            )
            rt_path.append(rt)

            _, margin_multiplier = system_aggregator.get_circuit_breaker_state(rt)

            im = base_im_whale * margin_multiplier

            if (whale_position_notional - pnl) < im:
                loss = im - (whale_position_notional - pnl)
                aes_default_fund -= loss
                if (initial_default_fund - aes_default_fund) > aes_max_drawdown:
                    aes_max_drawdown = initial_default_fund - aes_default_fund

            # --- FXD Logic ---
            if (whale_position_notional - pnl) < fxd_mm_level:
                loss = fxd_mm_level - (whale_position_notional - pnl)
                fxd_default_fund -= loss
                if (initial_default_fund - fxd_default_fund) > fxd_max_drawdown:
                    fxd_max_drawdown = initial_default_fund - fxd_default_fund

        aes_drawdowns.append(aes_max_drawdown / initial_default_fund)
        fxd_drawdowns.append(fxd_max_drawdown / initial_default_fund)
        if rt_path:
            rt_peaks.append(np.max(rt_path))

    return np.array(aes_drawdowns), np.array(fxd_drawdowns), np.array(rt_peaks)


def analyze_results(aes_drawdowns, fxd_drawdowns):
    """
    Calculates metrics and generates plots.
    """
    initial_default_fund = 50_000_000

    # --- Calculate Metrics ---
    pod_aes = np.sum(aes_drawdowns > 0.5) / len(aes_drawdowns)
    pod_fxd = np.sum(fxd_drawdowns > 0.5) / len(fxd_drawdowns)

    var_drawdown_aes = np.percentile(aes_drawdowns * initial_default_fund, 99)
    var_drawdown_fxd = np.percentile(fxd_drawdowns * initial_default_fund, 99)

    print("--- Analysis Results ---")
    print(f"AES Model PoD: {pod_aes:.2%}")
    print(f"FXD Model PoD: {pod_fxd:.2%}")
    print(f"AES Model 99% VaR of Drawdown: ${var_drawdown_aes:,.2f}")
    print(f"FXD Model 99% VaR of Drawdown: ${var_drawdown_fxd:,.2f}")

    # --- Generate Plots ---
    # Drawdown Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fxd_drawdowns, bins=50, alpha=0.7, label="FXD Model", color="red")
    plt.hist(aes_drawdowns, bins=50, alpha=0.7, label="AES Model", color="blue")
    plt.xlabel("Default Fund Drawdown (%)")
    plt.ylabel("Frequency")
    plt.title("Default Fund Drawdown Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig("drawdown_distribution.png")
    print("\nSaved drawdown distribution plot to drawdown_distribution.png")

    # PoD Comparison Bar Chart
    plt.figure(figsize=(8, 6))
    models = ["AES Model", "FXD Model"]
    pods = [pod_aes, pod_fxd]
    bars = plt.bar(models, pods, color=["blue", "red"])
    plt.ylabel("Probability of Default (PoD)")
    plt.title("PoD Comparison: AES vs. FXD")
    plt.grid(axis="y")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.2%}", va="bottom"
        )  # va is vertical alignment

    plt.savefig("pod_comparison.png")
    print("Saved PoD comparison plot to pod_comparison.png")


if __name__ == "__main__":
    aes_drawdowns, fxd_drawdowns, rt_peaks = run_montecarlo()
    analyze_results(aes_drawdowns, fxd_drawdowns)
