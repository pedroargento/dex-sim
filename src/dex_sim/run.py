import numpy as np

from simulation import run_simulation
from analysis import (
    summarize_df_requirement,
    summarize_defaults_and_haircuts,
)
from plotting import (
    plot_df_requirement_hist,
    plot_haircut_distribution,
    plot_default_times,
    plot_breaker_heatmap,
    plot_leverage_vs_price_move,
    plot_max_leverage_violin,
    plot_all_sample_paths,
)

if __name__ == "__main__":
    np.random.seed(42)

    sim = run_simulation(
        num_paths=10_000,
        initial_price=4000.0,
        total_oi=1_000_000_000.0,
        whale_oi_fraction=0.40,
        fxd_leverage=20.0,
        initial_default_fund=50_000_000.0,
        stress_factor=1.0,
        slippage_factor=0.001,
        num_sample_paths=5,
        garch_params_file="garch_params.json",
    )

    # Console summaries
    summarize_df_requirement(sim)
    summarize_defaults_and_haircuts(sim)

    # Distributions
    plot_df_requirement_hist(sim)
    plot_haircut_distribution(sim)

    # Path-level behaviour
    plot_all_sample_paths(sim, max_paths=5)

    # System-level diagnostics
    plot_default_times(sim)
    plot_breaker_heatmap(sim)
    plot_leverage_vs_price_move(sim)
    plot_max_leverage_violin(sim)
