import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_structures import SimulationResults


# ============================================================
# BASIC SETTINGS
# ============================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["font.size"] = 11


# ============================================================
# UTILITY: BREAKER SHADING
# ============================================================


def shade_breaker_regions(
    ax, breaker_state, ymin=None, ymax=None, alpha_soft=0.20, alpha_hard=0.25
):
    """
    Shade SOFT and HARD circuit-breaker periods on ax.
    breaker_state: array of shape (T,) with values {0,1,2}.
    """
    T = len(breaker_state)
    t = np.arange(T)

    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]

    soft = breaker_state == 1
    hard = breaker_state == 2

    ax.fill_between(
        t, ymin, ymax, where=soft, color="gold", alpha=alpha_soft, linewidth=0
    )
    ax.fill_between(
        t, ymin, ymax, where=hard, color="red", alpha=alpha_hard, linewidth=0
    )


# ============================================================
# 1. DF REQUIREMENT HISTOGRAM
# ============================================================


def plot_df_requirement_hist(sim: SimulationResults, filename="df_required_hist.png"):
    aes = sim.aes_df_required
    fxd = sim.fxd_df_required

    plt.figure(figsize=(10, 5))
    sns.histplot(aes, bins=50, kde=True, color="blue", label="AES")
    sns.histplot(fxd, bins=50, kde=True, color="red", label="FXD")
    plt.xlabel("DF Required (USD)")
    plt.ylabel("Frequency")
    plt.title("Default Fund Requirement Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 2. PRICE PATH
# ============================================================


def plot_price_path(sim: SimulationResults, k: int, filename=None):
    price = sim.price_paths[k]
    t = np.arange(len(price))

    plt.figure(figsize=(10, 4))
    plt.plot(t, price, color="black")
    plt.title(f"Sample Path {k} – Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    if filename is None:
        filename = f"sample_{k}_price.png"
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 3. LEVERAGE EVOLUTION (AES vs FXD)
# ============================================================


def plot_leverage_evolution(sim: SimulationResults, k: int, filename=None):
    t = np.arange(sim.horizon)
    lev_aes = sim.leverage_long_aes[k]
    lev_fxd = sim.leverage_long_fxd[k]

    plt.figure(figsize=(10, 4))
    plt.plot(t, lev_aes, label="AES", color="blue")
    plt.plot(t, lev_fxd, label="FXD", color="red")
    plt.title(f"Sample Path {k} – Long Leverage (AES vs FXD)")
    plt.xlabel("Time")
    plt.ylabel("Leverage")
    plt.legend()
    plt.grid(True)
    if filename is None:
        filename = f"sample_{k}_leverage_long.png"
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 4. Rₜ (Systemic Stress Index)
# ============================================================


def plot_systemic_Rt(sim: SimulationResults, k: int, filename=None):
    t = np.arange(sim.horizon)
    R_aes = sim.R_aes[k]
    R_fxd = sim.R_fxd[k]

    plt.figure(figsize=(10, 4))
    plt.plot(t, R_aes, label="AES", color="blue")
    plt.plot(t, R_fxd, label="FXD", color="red")
    plt.title(f"Sample Path {k} – Systemic Risk Index Rₜ")
    plt.xlabel("Time")
    plt.ylabel("Rₜ")
    plt.legend()
    plt.grid(True)
    if filename is None:
        filename = f"sample_{k}_Rt.png"
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 5. BREAKER STATE (step chart) with shading
# ============================================================


def plot_breaker_state(sim: SimulationResults, k: int, filename=None):
    t = np.arange(sim.horizon)
    br = sim.breaker_aes[k]  # AES breaker used for shading
    br_fxd = sim.breaker_fxd[k]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()

    # Draw AES breaker state (step)
    ax.step(t, br, where="post", color="blue", label="AES breaker")

    # Draw FXD breaker state (step)
    ax.step(t, br_fxd, where="post", color="red", label="FXD breaker")

    # Shading for AES breaker (soft/hard)
    shade_breaker_regions(ax, br, ymin=-0.5, ymax=2.5)

    plt.yticks([0, 1, 2], ["NORMAL", "SOFT", "HARD"])
    plt.title(f"Sample Path {k} – Breaker State")
    plt.xlabel("Time")
    plt.ylabel("Breaker")
    plt.grid(True)
    plt.legend()
    if filename is None:
        filename = f"sample_{k}_breaker.png"
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 6. PATH ANATOMY: Multi-axis combined chart
#    Price + Leverage + Rₜ + Breaker shading
# ============================================================


def plot_path_anatomy(sim: SimulationResults, k: int, filename=None):
    t = np.arange(sim.horizon)
    price = sim.price_paths[k]
    lev_aes = sim.leverage_long_aes[k]
    lev_fxd = sim.leverage_long_fxd[k]
    R_aes = sim.R_aes[k]
    breaker = sim.breaker_aes[k]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # PRICE
    ax1.plot(t, price, color="black", label="Price")
    ax1.set_ylabel("Price", color="black")

    # LEVERAGE
    ax1.plot(t, lev_aes, color="blue", alpha=0.7, label="AES leverage")
    ax1.plot(t, lev_fxd, color="red", alpha=0.7, label="FXD leverage")

    # Rₜ on second axis
    ax2 = ax1.twinx()
    ax2.plot(t, R_aes, color="purple", linestyle="--", label="AES Rₜ")
    ax2.set_ylabel("Rₜ", color="purple")

    # Breaker shading
    shade_breaker_regions(ax1, breaker)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"Sample Path {k} – Path Anatomy")
    plt.xlabel("Time")
    plt.grid(True)

    if filename is None:
        filename = f"sample_{k}_anatomy.png"
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 7. HAIRCUT DISTRIBUTION
# ============================================================


def plot_haircut_distribution(
    sim: SimulationResults, filename="haircut_distribution.png"
):
    aes_hc = sim.aes_haircuts
    fxd_hc = sim.fxd_haircuts

    plt.figure(figsize=(9, 5))
    if np.any(aes_hc > 0):
        sns.kdeplot(aes_hc[aes_hc > 0], label="AES", color="blue", shade=True)
    if np.any(fxd_hc > 0):
        sns.kdeplot(fxd_hc[fxd_hc > 0], label="FXD", color="red", shade=True)

    plt.title("Haircut Distribution (Fraction of VM Not Paid)")
    plt.xlabel("Haircut Fraction")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 8. DEFAULT TIMING HISTOGRAM
# ============================================================


def plot_default_times(sim: SimulationResults, filename="default_timing.png"):
    aes = sim.aes_defaults
    fxd = sim.fxd_defaults

    aes_times = np.where(aes)[0]
    fxd_times = np.where(fxd)[0]

    plt.figure(figsize=(9, 5))
    sns.histplot(aes_times, color="blue", label="AES", bins=40)
    sns.histplot(fxd_times, color="red", label="FXD", bins=40)
    plt.title("Default Timing Distribution")
    plt.xlabel("Path index (ordered by occurrence)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 9. BREAKER HEATMAP (paths × time)
# ============================================================


def plot_breaker_heatmap(sim: SimulationResults, filename="breaker_heatmap.png"):
    mat = sim.breaker_aes  # shape [sample_paths, T]
    plt.figure(figsize=(10, 5))
    sns.heatmap(mat, cmap=["white", "gold", "red"], cbar=False)
    plt.title("AES Breaker Heatmap (sample paths × time)")
    plt.xlabel("Time")
    plt.ylabel("Sample path index")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 10. LEVERAGE vs PRICE MOVE HEXBIN (diagnostic)
# ============================================================


def plot_leverage_vs_price_move(
    sim: SimulationResults, filename="hex_leverage_price.png"
):
    # Flatten sample paths
    lr = np.diff(sim.price_paths, axis=1) / sim.price_paths[:, :-1]  # simple return
    lev = sim.leverage_long_aes[:, :-1]

    lr_flat = lr.flatten()
    lev_flat = lev.flatten()

    plt.figure(figsize=(8, 6))
    plt.hexbin(lr_flat, lev_flat, gridsize=60, cmap="viridis")
    plt.colorbar(label="Count")
    plt.xlabel("Price Return")
    plt.ylabel("AES Leverage")
    plt.title("Leverage vs Price Move (AES)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 11. VIOLIN PLOT OF MAX LEVERAGE
# ============================================================


def plot_max_leverage_violin(
    sim: SimulationResults, filename="max_leverage_violin.png"
):
    max_aes = np.nanmax(sim.leverage_long_aes, axis=1)
    max_fxd = np.nanmax(sim.leverage_long_fxd, axis=1)

    data = [max_aes, max_fxd]

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=data, palette=["blue", "red"])
    plt.xticks([0, 1], ["AES", "FXD"])
    plt.ylabel("Max Leverage Across Path")
    plt.title("Distribution of Maximum Leverage")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[saved] {filename}")


# ============================================================
# 12. RUN ALL PATH CHARTS FOR N SAMPLE PATHS
# ============================================================


def plot_all_sample_paths(sim: SimulationResults, max_paths=5):
    for k in range(min(max_paths, len(sim.sample_idx))):
        plot_price_path(sim, k)
        plot_leverage_evolution(sim, k)
        plot_systemic_Rt(sim, k)
        plot_breaker_state(sim, k)
        plot_path_anatomy(sim, k)
