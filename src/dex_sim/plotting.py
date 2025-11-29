"""
plotting.py — Visualization layer for dex-sim results.

Works with SingleModelResults and MultiModelResults.
Handles optional fields like R_t, breaker_state, margin_multiplier,
partial liquidation amounts, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------------------------------------
# High-level functions
# ----------------------------------------------------------------------


def plot_all_for_model(model_res, outdir="plots", max_paths=5):
    """
    Generate all relevant plots for a single model.
    Saves them in outdir/<model_name>/...
    """
    d = os.path.join(outdir, model_res.name)
    _ensure_dir(d)

    plot_df_requirement_hist(model_res, d)
    plot_price_paths(model_res, d, max_paths)
    plot_leverage_paths(model_res, d, max_paths)

    if model_res.rt is not None:
        plot_rt_paths(model_res, d, max_paths)
        plot_rt_distribution(model_res, d)

    if model_res.breaker_state is not None:
        plot_breaker_heatmap(model_res, d)
        plot_breaker_transition_probs(model_res, d)

    if model_res.margin_multiplier is not None:
        plot_margin_multiplier_paths(model_res, d, max_paths)
        plot_margin_multiplier_distribution(model_res, d)

    # Optional: HL partial liquidation
    if model_res.partial_liq_amount is not None:
        plot_partial_liquidation(model_res, d, max_paths)

    if model_res.equity_long is not None:
        plot_equity_paths(model_res, d, max_paths)


# ----------------------------------------------------------------------
# Distribution: Default Fund requirement
# ----------------------------------------------------------------------


def plot_df_requirement_hist(model_res, outdir="plots"):
    df = model_res.df_required
    plt.figure(figsize=(8, 5))
    sns.histplot(df, bins=50, kde=False)
    plt.title(f"DF Requirement Distribution — {model_res.name}")
    plt.xlabel("DF Required ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_df_hist.png"))
    plt.close()


# ----------------------------------------------------------------------
# Price paths
# ----------------------------------------------------------------------


def plot_price_paths(model_res, outdir, max_paths=5):
    P, T = model_res.price_paths.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(model_res.price_paths[i], alpha=0.7)
    plt.title(f"Sample Price Paths — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_price_paths.png"))
    plt.close()


# ----------------------------------------------------------------------
# Leverage paths
# ----------------------------------------------------------------------


def plot_leverage_paths(model_res, outdir, max_paths=5):
    P, T = model_res.lev_long.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(model_res.lev_long[i], label=f"path {i}", alpha=0.6)
    plt.title(f"Long Leverage (sample paths) — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Leverage (Long)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_long_lev.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(model_res.lev_short[i], alpha=0.6)
    plt.title(f"Short Leverage (sample paths) — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Leverage (Short)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_short_lev.png"))
    plt.close()


# ----------------------------------------------------------------------
# R_t (systemic risk)
# ----------------------------------------------------------------------


def plot_rt_paths(model_res, outdir, max_paths=5):
    if model_res.rt is None:
        return

    P, T = model_res.rt.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(model_res.rt[i], alpha=0.7)
    plt.title(f"R_t (Systemic Stress) — sample paths — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("R_t")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_rt_paths.png"))
    plt.close()


def plot_rt_distribution(model_res, outdir):
    if model_res.rt is None:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(model_res.rt.flatten(), bins=60)
    plt.title(f"R_t Distribution — {model_res.name}")
    plt.xlabel("R_t")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_rt_dist.png"))
    plt.close()


# ----------------------------------------------------------------------
# Breaker state visualizations
# ----------------------------------------------------------------------


def plot_breaker_heatmap(model_res, outdir):
    if model_res.breaker_state is None:
        return

    plt.figure(figsize=(12, 5))
    plt.imshow(model_res.breaker_state, aspect="auto", cmap="viridis")
    plt.colorbar(label="Breaker State (0=NORMAL,1=SOFT,2=HARD)")
    plt.title(f"Breaker State Heatmap — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Path")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_breaker_heatmap.png"))
    plt.close()


def plot_breaker_transition_probs(model_res, outdir):
    if model_res.breaker_state is None:
        return

    bs = model_res.breaker_state
    P, T = bs.shape

    # compute transitions 0->1, 1->2, 2->2, etc.
    transitions = np.zeros((3, 3))
    for p in range(P):
        for t in range(1, T):
            transitions[bs[p, t - 1], bs[p, t]] += 1

    # normalize row-wise
    row_sums = transitions.sum(axis=1, keepdims=True)
    probs = np.divide(transitions, row_sums, where=row_sums > 0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        probs,
        annot=True,
        cmap="Blues",
        xticklabels=["N", "S", "H"],
        yticklabels=["N", "S", "H"],
        fmt=".2f",
    )
    plt.title(f"Breaker Transition Probabilities — {model_res.name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_breaker_transitions.png"))
    plt.close()


# ----------------------------------------------------------------------
# Margin multiplier
# ----------------------------------------------------------------------


def plot_margin_multiplier_paths(model_res, outdir, max_paths=5):
    if model_res.margin_multiplier is None:
        return

    P, T = model_res.margin_multiplier.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.step(np.arange(T), model_res.margin_multiplier[i], where="mid", alpha=0.6)
    plt.title(f"Margin Multiplier — sample paths — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Multiplier")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_margin_mult_paths.png"))
    plt.close()


def plot_margin_multiplier_distribution(model_res, outdir):
    if model_res.margin_multiplier is None:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(model_res.margin_multiplier.flatten(), bins=20)
    plt.title(f"Margin Multiplier Distribution — {model_res.name}")
    plt.xlabel("Multiplier")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_margin_mult_dist.png"))
    plt.close()


# ----------------------------------------------------------------------
# Equity paths (optional)
# ----------------------------------------------------------------------


def plot_equity_paths(model_res, outdir, max_paths=5):
    if model_res.equity_long is None:
        return

    P, T = model_res.equity_long.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(model_res.equity_long[i], alpha=0.7)
    plt.title(f"Equity (Long) — sample paths — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_equity_long.png"))
    plt.close()

    if model_res.equity_short is not None:
        plt.figure(figsize=(10, 5))
        for i in range(paths):
            plt.plot(model_res.equity_short[i], alpha=0.7)
        plt.title(f"Equity (Short) — sample paths — {model_res.name}")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{model_res.name}_equity_short.png"))
        plt.close()


# ----------------------------------------------------------------------
# Partial liquidation (Hyperliquid-like)
# ----------------------------------------------------------------------


def plot_partial_liquidation(model_res, outdir, max_paths=5):
    if model_res.partial_liq_amount is None:
        return

    P, T = model_res.partial_liq_amount.shape
    paths = min(P, max_paths)

    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.step(np.arange(T), model_res.partial_liq_amount[i], where="post", alpha=0.6)
    plt.title(f"Partial Liquidation Amount — sample paths — {model_res.name}")
    plt.xlabel("Time")
    plt.ylabel("Notional Closed")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_partial_liq.png"))
    plt.close()
