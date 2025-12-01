"""
plotting.py — Professional Visualization Suite for Dex-Sim Risk Engine.

This module provides a comprehensive set of visualizations to analyze:
1. Solvency & Tail Risk (Survival curves, Violin plots)
2. Systemic Dynamics (Regime switching, R_t, Breakers)
3. Microstructure Mechanics (Liquidation heatmaps, Slippage waterfalls)
4. Convergence & Stability (Monte Carlo confidence)

Designed for use with Matplotlib and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Optional, Tuple

from .data_structures import MultiModelResults, SingleModelResults


# ==============================================================================
#  Configuration & Helpers
# ==============================================================================

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = sns.color_palette("deep")


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def _get_worst_path_idx(model_res: SingleModelResults) -> int:
    """Returns the index of the path with the highest DF requirement."""
    return int(np.argmax(model_res.df_required))


# ==============================================================================
#  1. Solvency & Tail Risk Dashboards
# ==============================================================================


def plot_df_survival_curve(results: MultiModelResults, outdir: str):
    """
    Log-Log Survival Function (Reverse CDF) of Default Fund Usage. 
    
    Insight:
        Reveals power-law tails in loss distributions. Straight lines on this 
        log-log plot indicate heavy tails (infinite variance potential).
        Comparing models here shows which one cuts off extreme tail risks better.
    """
    plt.figure(figsize=(10, 6))
    
    has_data = False
    for name, model_res in results.models.items():
        # Filter to positive losses to show the tail structure clearly
        data = np.sort(model_res.df_required)
        data = data[data > 0]
        if len(data) == 0:
            continue
        
        has_data = True
        n = len(data)
        # Survival probability P(X > x)
        y = np.arange(n, 0, -1) / len(model_res.df_required) # Normalize by TOTAL paths
        
        plt.plot(data, y, lw=2.5, label=name)

    if not has_data:
        plt.close()
        return

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("DF Usage Amount ($) [Log]")
    plt.ylabel("Probability P(Loss > x) [Log]")
    plt.title("Solvency Survival Curve (Tail Risk Analysis)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solvency_survival_curve.png"), dpi=150)
    plt.close()


def plot_model_comparison_violins(results: MultiModelResults, outdir: str):
    """
    Split Violin Plot comparing DF Usage distributions across models.
    
    Insight:
        Visualizes the density of losses. A wider base means frequent small losses.
        Long thin necks mean rare catastrophic losses.
    """
    data_list = []
    for name, model_res in results.models.items():
        # We include zeros? Usually better to inspect non-zero losses for shape,
        # or all data for overall density. Let's use non-zero for clarity of risk events.
        losses = model_res.df_required[model_res.df_required > 1.0] # Filter trivial noise
        if len(losses) > 0:
            df_temp = pd.DataFrame({"Model": name, "DF Usage": losses})
            data_list.append(df_temp)
    
    if not data_list:
        return

    df_all = pd.concat(data_list)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_all, 
        x="Model", 
        y="DF Usage", 
        cut=0, 
        scale="width", 
        palette="muted",
        inner="box"
    )
    plt.title("Distribution of Material Default Fund Usage (> $1)")
    plt.ylabel("DF Usage ($)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "df_distribution_violins.png"), dpi=150)
    plt.close()


def plot_efficiency_frontier(results: MultiModelResults, outdir: str):
    """
    Scatter plot of Safety (Max DF Usage) vs Efficiency (Average Margin).
    
    Insight:
        Demonstrates the cost-benefit trade-off. The ideal model is in the 
        bottom-left (Low DF usage, Low Margin).
    """
    plt.figure(figsize=(10, 6))
    
    for name, model_res in results.models.items():
        # X: Efficiency (Mean Margin Requirement relative to Notional?)
        # We can approx mean leverage or mean margin multiplier.
        # Let's use Mean Margin Multiplier * IM0 (approx).
        # Better: Average Equity / Notional over time?
        # Let's use simple Mean Margin Multiplier if available, else 1.0
        
        if model_res.margin_multiplier is not None:
            avg_mult = np.mean(model_res.margin_multiplier)
        else:
            avg_mult = 1.0
            
        # Y: Safety (99.9% DF Usage or Max)
        max_loss = np.max(model_res.df_required)
        p99_loss = np.percentile(model_res.df_required, 99.9)
        
        plt.scatter(avg_mult, p99_loss, s=100, label=name, alpha=0.8, edgecolors='w')
        plt.text(avg_mult, p99_loss, f"  {name}", fontsize=9, va='center')

    plt.xlabel("Average Margin Multiplier (Capital Inefficiency)")
    plt.ylabel("99.9% DF Requirement ($)")
    plt.title("Safety vs. Efficiency Frontier")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "efficiency_frontier.png"), dpi=150)
    plt.close()


# ==============================================================================
#  2. Systemic Dynamics & Regime Switching
# ==============================================================================


def plot_regime_dynamics(model_res: SingleModelResults, outdir: str):
    """
    Composite time-series showing the causal chain:
    Volatility -> R_t -> Breaker State -> Margin Multipliers.
    
    Insight:
        Validates the feedback loop mechanism. Shows if the breaker triggers
        too late (after price crash) or pre-emptively.
    """
    if model_res.rt is None or model_res.breaker_state is None:
        return

    # Use the worst case path for "stress" visualization
    idx = _get_worst_path_idx(model_res)
    T = model_res.rt.shape[1]
    t_axis = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Top: Price & Risk Index ---
    ax0 = axes[0]
    ax0.plot(t_axis, model_res.price_paths[idx], color='#2c3e50', lw=1.5, label='Asset Price')
    ax0.set_ylabel('Price ($)', color='#2c3e50', fontweight='bold')
    ax0.tick_params(axis='y', labelcolor='#2c3e50')
    ax0.grid(True, alpha=0.2)
    
    ax0t = ax0.twinx()
    ax0t.plot(t_axis, model_res.rt[idx], color='#e74c3c', lw=1.5, ls='--', label='Risk Index ($R_t$)')
    ax0t.set_ylabel('$R_t$ Index', color='#e74c3c', fontweight='bold')
    ax0t.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add legend combining both
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0t.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc='upper left')
    ax0.set_title(f"Regime Dynamics Autopsy (Path #{idx}) — {model_res.name}")

    # --- Middle: Breaker State ---
    ax1 = axes[1]
    bs = model_res.breaker_state[idx]
    
    # Background shading for states
    ax1.fill_between(t_axis, 0, 1, where=(bs==0), color='green', alpha=0.1, transform=ax1.get_xaxis_transform())
    ax1.fill_between(t_axis, 0, 1, where=(bs==1), color='orange', alpha=0.2, transform=ax1.get_xaxis_transform())
    ax1.fill_between(t_axis, 0, 1, where=(bs==2), color='red', alpha=0.3, transform=ax1.get_xaxis_transform())
    
    ax1.step(t_axis, bs, where='post', color='black', lw=1)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['NORMAL', 'SOFT', 'HARD'])
    ax1.set_ylabel("Breaker State")
    ax1.grid(False)

    # --- Bottom: Margin Multiplier ---
    ax2 = axes[2]
    if model_res.margin_multiplier is not None:
        ax2.step(t_axis, model_res.margin_multiplier[idx], where='post', color='blue', lw=1.5)
    ax2.set_ylabel("Margin Multiplier")
    ax2.set_xlabel("Simulation Time Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_regime_dynamics.png"), dpi=150)
    plt.close()


# ==============================================================================
#  3. Liquidation Mechanics & Heatmaps
# ==============================================================================


def plot_liquidation_heatmap(model_res: SingleModelResults, outdir: str):
    """
    Heatmap of Liquidation Fraction (k) across paths and time.
    Sorted by total DF usage (worst paths at top).
    
    Insight:
        Reveals 'Death Spirals' (vertical bands) vs idiosyncratic failures.
        Shows gradient of partial liquidation vs hard closeouts (red).
    """
    if model_res.liquidation_fraction is None:
        return

    # Sort paths by severity (DF Usage)
    severity_idx = np.argsort(model_res.df_required) # Ascending
    # We want worst at top
    sorted_liq = model_res.liquidation_fraction[severity_idx]
    
    # If too many paths, downsample rows for visualization
    max_rows = 1000
    if sorted_liq.shape[0] > max_rows:
        # Pick top N worst + a sample of others
        top_n = 800
        rest_n = 200
        worst = sorted_liq[-top_n:]
        rest = sorted_liq[:-top_n:int((len(sorted_liq)-top_n)/rest_n)]
        plot_data = np.vstack([rest, worst])
        ylabel = f"Paths (Top {top_n} Worst + Sample)"
    else:
        plot_data = sorted_liq
        ylabel = "Paths (Sorted by Loss)"

    plt.figure(figsize=(12, 7))
    cmap = sns.color_palette("Reds", as_cmap=True)
    
    # Log scale color or simple linear? Linear is fine if k is 0..1.
    # Mask 0 for white background
    mask = plot_data == 0
    
    sns.heatmap(plot_data, cmap=cmap, mask=mask, cbar_kws={'label': 'Liquidation Fraction ($k$)'},
                xticklabels=500, yticklabels=False, vmin=0, vmax=1)
    
    plt.title(f"Liquidation Intensity Heatmap — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_liquidation_heatmap.png"), dpi=150)
    plt.close()

def plot_notional_fan_chart(model_res: SingleModelResults, outdir: str):
    """
    Fan Chart showing the decay of Open Interest (Notional) over time.
    
    Insight:
        Shows the system's deleveraging speed. Does liquidity dry up instantly
        (vertical drop) or does the system manage a soft landing?
    """
    if model_res.notional_paths is None:
        return
        
    paths = model_res.notional_paths
    t_axis = np.arange(paths.shape[1])
    
    # Percentiles
    p05 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, p50, color='navy', lw=2, label='Median Notional')
    
    # Fans
    plt.fill_between(t_axis, p25, p75, color='tab:blue', alpha=0.4, label='IQR (25-75%)')
    plt.fill_between(t_axis, p05, p95, color='tab:blue', alpha=0.15, label='5-95% Range')
    
    plt.title(f"Systemic De-leveraging (Notional Decay) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Remaining Open Interest ($)")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_notional_fan.png"), dpi=150)
    plt.close()

def plot_slippage_waterfall(model_res: SingleModelResults, outdir: str):
    """
    Breakdown of Default Fund Usage: Slippage vs. Bankruptcy (Negative Equity).
    
    Insight:
        Diagnoses the SOURCE of loss.
        - High Slippage -> Liquidation algo is too aggressive or liquidity too low.
        - High Bankruptcy -> Margins are too low (gap risk).
    """
    if model_res.slippage_cost is None:
        return

    total_df = np.sum(model_res.df_required)
    total_slippage = np.sum(model_res.slippage_cost) # Note: this includes user-paid slippage?
    # engine.py: df_required += cost (if shortfall).
    # Ideally we strictly sum df_path components if we had them separated.
    # Let's assume `slippage_cost` array tracks ALL slippage generated.
    # We can't perfectly separate USER paid vs DF paid slippage without more engine outputs.
    # Approximation: DF_Slippage ~ min(DF_Required, Total_Slippage) for that path?
    
    # Let's use aggregates over all paths for a global view.
    # Total Slippage Generated vs Total DF Used. 
    
    # Actually, chart G logic: Stacked Bar of Average per path. 
    
    avg_df = np.mean(model_res.df_required)
    avg_slippage_gen = np.mean(np.sum(model_res.slippage_cost, axis=1))
    
    # This is tricky without precise accounting.
    # Let's plot the distributions side-by-side.
    
    data = pd.DataFrame({
        'Metric': ['DF Required', 'Slippage Generated'],
        'Value': [avg_df, avg_slippage_gen]
    })
    
    plt.figure(figsize=(7, 6))
    sns.barplot(data=data, x='Metric', y='Value', palette="viridis")
    plt.title(f"Average Cost Composition — {model_res.name}")
    plt.ylabel("Amount ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_cost_composition.png"), dpi=150)
    plt.close()


# ==============================================================================
#  4. Path Forensics (Autopsy)
# ==============================================================================


def plot_worst_case_autopsy(model_res: SingleModelResults, outdir: str):
    """
    Detailed anatomy of the single worst loss event in the simulation.
    
    Insight:
        Tells the narrative of the failure. Did they bleed out slowly (partial liq)
        or die instantly (gap)?
    """
    idx = _get_worst_path_idx(model_res)
    T = model_res.price_paths.shape[1]
    t = np.arange(T)
    
    # Create composite layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # 1. Price and Liquidation Events
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, model_res.price_paths[idx], color='black', alpha=0.6, label='Price')
    
    # Mark liquidations
    if model_res.liquidation_fraction is not None:
        liqs = model_res.liquidation_fraction[idx]
        liq_events = np.where(liqs > 0)[0]
        if len(liq_events) > 0:
            # Scatter size proportional to k
            sizes = liqs[liq_events] * 100
            ax0.scatter(t[liq_events], model_res.price_paths[idx][liq_events], 
                        color='red', s=sizes, label='Liquidation', zorder=5)
    
    ax0.set_title(f"Worst Case Autopsy (Path #{idx}) — {model_res.name}\nTotal DF Used: ${model_res.df_required[idx]:,.2f}")
    ax0.set_ylabel("Price ($)")
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    # 2. Trader Equity vs Margin Req (approx)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    if model_res.equity_long is not None:
        ax1.plot(t, model_res.equity_long[idx], label='Long Equity', color='green')
    if model_res.equity_short is not None:
        ax1.plot(t, model_res.equity_short[idx], label='Short Equity', color='red')
    
    ax1.axhline(0, color='black', lw=1.5)
    # Shade bankruptcy
    if model_res.equity_long is not None:
        ax1.fill_between(t, 0, model_res.equity_long[idx], where=(model_res.equity_long[idx]<0), color='black', alpha=0.5)
        ax1.fill_between(t, 0, model_res.equity_short[idx], where=(model_res.equity_short[idx]<0), color='black', alpha=0.5)
        
    ax1.set_ylabel("Trader Equity ($)")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # 3. Cumulative DF Usage
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    if model_res.df_path is not None:
        cumulative_df = np.cumsum(model_res.df_path[idx])
        ax2.fill_between(t, 0, cumulative_df, color='darkred', alpha=0.6)
        ax2.plot(t, cumulative_df, color='red')
    else:
        # Fallback if path not available
        ax2.text(0.5, 0.5, "DF Path data not available", ha='center')
        
    ax2.set_ylabel("Cumul. DF Usage ($)")
    ax2.set_xlabel("Time Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_autopsy.png"), dpi=150)
    plt.close()


# ==============================================================================
#  5. Convergence & Stability
# ==============================================================================


def plot_mc_convergence(results: MultiModelResults, outdir: str):
    """
    Checks statistical stability of the simulation.
    Plots running average of DF Usage as paths increase.
    """
    plt.figure(figsize=(10, 5))
    
    for name, model_res in results.models.items():
        # Running mean
        df_req = model_res.df_required
        cumulative_sum = np.cumsum(df_req)
        counts = np.arange(1, len(df_req) + 1)
        running_mean = cumulative_sum / counts
        
        plt.plot(counts, running_mean, label=name)
        
        # Add Error Bands (Standard Error)
        # Std dev of sample / sqrt(N)
        # Calculating running std is expensive, assume convergence if mean stabilizes
    
    plt.xlabel("Number of Paths")
    plt.ylabel("Average DF Usage ($)")
    plt.title("Monte Carlo Convergence Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mc_convergence.png"), dpi=150)
    plt.close()


# ==============================================================================
#  6. System Leverage Analysis
# ==============================================================================


def plot_system_leverage(model_res: SingleModelResults, outdir: str):
    """
    Visualizes aggregate system leverage for Long vs Short sides over time.
    
    Insight:
        - Tracks how leverage spikes during market moves.
        - Shows if AES effectively caps leverage compared to FXD.
        - Highlights the asymmetry between Long/Short leverage during trends.
    """
    # Check if we have leverage data
    if model_res.lev_long is None or model_res.lev_short is None:
        return

    # Arrays are [Paths, Time]. We want statistics over Paths.
    # Use nanmean/nanpercentile because failed paths might have NaNs.
    
    t_axis = np.arange(model_res.lev_long.shape[1])
    
    # Long Side Stats
    l_mean = np.nanmean(model_res.lev_long, axis=0)
    l_p05 = np.nanpercentile(model_res.lev_long, 5, axis=0)
    l_p95 = np.nanpercentile(model_res.lev_long, 95, axis=0)

    # Short Side Stats
    s_mean = np.nanmean(model_res.lev_short, axis=0)
    s_p05 = np.nanpercentile(model_res.lev_short, 5, axis=0)
    s_p95 = np.nanpercentile(model_res.lev_short, 95, axis=0)

    plt.figure(figsize=(10, 6))
    
    # Long Plot (Blue)
    plt.plot(t_axis, l_mean, color='tab:blue', lw=2, label='Long Leverage (Mean)')
    plt.fill_between(t_axis, l_p05, l_p95, color='tab:blue', alpha=0.15, label='Long 5-95% Range')

    # Short Plot (Red)
    plt.plot(t_axis, s_mean, color='tab:red', lw=2, label='Short Leverage (Mean)')
    plt.fill_between(t_axis, s_p05, s_p95, color='tab:red', alpha=0.15, label='Short 5-95% Range')

    # Optional: Overlay Breaker States (Mode)
    if model_res.breaker_state is not None:
        # Calculate mode of breaker state at each timestep
        # scipy.stats.mode is robust but heavy. Let's just check mean > threshold for visual indication?
        # Or just: if > 50% of paths are in HARD state, shade it.
        
        # Fraction of paths in Hard State (2)
        frac_hard = np.mean(model_res.breaker_state == 2, axis=0)
        # Shade regions where > 20% of paths are Hard
        mask_hard = frac_hard > 0.2
        plt.fill_between(t_axis, 0, plt.ylim()[1], where=mask_hard, 
                         color='gray', alpha=0.1, transform=plt.gca().get_xaxis_transform(), 
                         label='High Stress Regime (>20% Breakers)')

    plt.title(f"Total System Leverage Over Time (Long vs Short) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Effective Leverage (x)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_system_leverage.png"), dpi=150)
    plt.close()


def plot_oi_with_breaker_bands(model_res: SingleModelResults, outdir: str):
    """
    Stacked area chart of OI (Long/Short) over time with Breaker State bands.
    Focuses on the path with the highest DF usage (stress test).
    
    Insight:
        - Shows if OI contracts when Breaker enters HARD mode (Red).
        - Correlates market size with risk regime.
    """
    if model_res.notional_paths is None or model_res.breaker_state is None:
        return
    
    idx = _get_worst_path_idx(model_res)
    
    # Data
    T = model_res.notional_paths.shape[1]
    t = np.arange(T)
    oi_side = model_res.notional_paths[idx]
    states = model_res.breaker_state[idx]
    
    plt.figure(figsize=(12, 6))
    
    # Stackplot
    # Assuming symmetry (OI Long ~ OI Short ~ Notional Per Side) for now
    # Total OI = Long + Short
    plt.stackplot(t, oi_side, oi_side, labels=['Long OI', 'Short OI'], 
                  colors=['#2980b9', '#c0392b'], alpha=0.7)
    
    # TODO: Overlay Whale Concentration Line
    # whale_share = calculate_whale_share(...)
    # plt.plot(t, whale_share * (oi_side * 2), color='gold', lw=1.5, label='Whale Concentration (Top 1%)')
    
    # Breaker Bands
    # Use axes transform to fill vertical height
    ax = plt.gca()
    trans = ax.get_xaxis_transform()
    
    # Normal (0) = Green, Soft (1) = Orange, Hard (2) = Red
    ax.fill_between(t, 0, 1, where=(states==0), color='green', alpha=0.1, transform=trans, label='Normal')
    ax.fill_between(t, 0, 1, where=(states==1), color='orange', alpha=0.2, transform=trans, label='Soft')
    ax.fill_between(t, 0, 1, where=(states==2), color='red', alpha=0.3, transform=trans, label='Hard')

    plt.title(f"OI Timeline with Breaker Bands (Path #{idx}) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Open Interest ($)")
    
    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_oi_breaker_bands.png"), dpi=150)
    plt.close()


def plot_systemic_solvency_curve(model_res: SingleModelResults, outdir: str):
    """
    Dual-Line Chart comparing Total System Equity vs Total Maintenance Margin.
    
    Insight:
        - Visualizes the system's "Safety Buffer".
        - Green Fill: Solvent (Equity > MM).
        - Red Fill: Insolvent/Fragile (Equity < MM).
    """
    if model_res.equity_long is None or model_res.equity_short is None:
        return
    
    idx = _get_worst_path_idx(model_res)
    t = np.arange(model_res.equity_long.shape[1])
    
    # Total Equity (System-wide)
    total_equity = model_res.equity_long[idx] + model_res.equity_short[idx]
    
    # Total MM Required (Approximation based on available data)
    # MM ~ Notional * Base_IM_Rate * Breaker_Mult * Gamma(0.8)
    # We need Base_IM_Rate. Let's infer from t=0 state or metadata.
    # Fallback: Use Lev_Long and Lev_Short to infer IM usage? 
    # Or just use: Margin Multiplier * (Notional / Base_Leverage)?
    # Let's assume a standard Base IM of 10% (10x) if not known, or try to reverse engineer.
    # Better: Use (Equity / Leverage) * Gamma? No, Equity != Margin Locked.
    # Best available proxy: Notional / Initial_Leverage * Multiplier * 0.8
    # We don't have Initial_Leverage easily.
    # Alternative: Use `df_required` spikes to pinpoint MM breaches? No.
    # Let's use a heuristic: MM_Total approx 5-10% of Notional * Multiplier.
    # If metadata has 'im_rate', use it.
    
    # Heuristic for Plotting (Visual Guide Only):
    # Assume Base MM is ~5% of Notional (20x max lev).
    # Adjust dynamically by breaker multiplier.
    base_mm_rate = 0.05 
    if model_res.margin_multiplier is not None:
        mm_mult = model_res.margin_multiplier[idx]
    else:
        mm_mult = np.ones_like(t)
        
    if model_res.notional_paths is not None:
        # Total Notional = Long + Short ~ 2 * Side Notional
        total_notional = model_res.notional_paths[idx] * 2.0 
        total_mm = total_notional * base_mm_rate * mm_mult
    else:
        return

    plt.figure(figsize=(10, 6))
    
    plt.plot(t, total_equity, color='#27ae60', lw=2, label='Total Trader Equity')
    plt.plot(t, total_mm, color='#c0392b', lw=2, ls='--', label='Total MM Required (Est.)')
    
    # Fill
    plt.fill_between(t, total_equity, total_mm, where=(total_equity >= total_mm), 
                     color='green', alpha=0.1, interpolate=True, label='Solvent Buffer')
    plt.fill_between(t, total_equity, total_mm, where=(total_equity < total_mm), 
                     color='red', alpha=0.2, interpolate=True, label='Insolvency Risk')
    
    plt.title(f"Systemic Solvency Curve (Health Monitor) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Value ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_solvency_curve.png"), dpi=150)
    plt.close()


def plot_trade_gating_waterfall(model_res: SingleModelResults, outdir: str):
    """
    Stacked Bar Chart of Trade Intent Gating (Waterall).
    Shows volume Accepted (Normal), Accepted (Risk-Reducing), and Rejected.
    
    Insight:
        - "How much flow was blocked by the breaker?"
        - "Did reduce-only mode allow exits while blocking entries?"
    """
    if model_res.notional_paths is None:
        return

    idx = _get_worst_path_idx(model_res)
    T = model_res.notional_paths.shape[1]
    t = np.arange(T - 1) # Diff reduces length by 1
    
    # --- Placeholder Data Logic (TODO: Wire to engine outputs) ---
    # We use Net Notional Change as a proxy for "Accepted Normal" volume.
    # Real engine should export: intent_accepted_normal, intent_accepted_reduce, intent_rejected
    
    notional = model_res.notional_paths[idx]
    delta_oi = np.abs(np.diff(notional)) # Proxy: Absolute volume change
    
    accepted_normal = delta_oi
    accepted_reduce = np.zeros_like(delta_oi)
    rejected = np.zeros_like(delta_oi)
    
    # If Breaker State is available, we can pretend some volume was "Reduce Only"
    if model_res.breaker_state is not None:
        states = model_res.breaker_state[idx][1:] # Align with diff
        # Mock logic: In Hard Mode (2), treat all flow as "Accepted Reduce" (visual test)
        mask_hard = (states == 2)
        accepted_reduce[mask_hard] = accepted_normal[mask_hard]
        accepted_normal[mask_hard] = 0
        
        # Mock logic: Random "Rejected" volume in Hard Mode to show the red bar
        # rejected[mask_hard] = accepted_reduce[mask_hard] * 0.5 
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    p1 = plt.bar(t, accepted_normal, color='#27ae60', width=1.0, alpha=0.8, label='Accepted (Normal)')
    p2 = plt.bar(t, accepted_reduce, bottom=accepted_normal, color='#f1c40f', width=1.0, alpha=0.8, label='Accepted (Reduce-Only)')
    p3 = plt.bar(t, rejected, bottom=accepted_normal+accepted_reduce, color='#c0392b', width=1.0, alpha=0.6, label='Rejected (Gated)')
    
    # Annotation
    plt.text(0.5, 0.95, "TODO: Integration with engine intent logs pending.\n(Showing proxy data based on OI change)", 
             transform=plt.gca().transAxes, ha='center', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title(f"Trade Intent Gating Waterfall — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Volume / Notional Change ($)")
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_trade_gating_waterfall.png"), dpi=150)
    plt.close()


def plot_enhanced_liquidation_timeline(model_res: SingleModelResults, outdir: str):
    """
    Price Chart Overlay with granular liquidation markers.
    Distinguishes between Partial (Soft) and Full (Hard) liquidations.
    """
    idx = _get_worst_path_idx(model_res)
    
    price = model_res.price_paths[idx]
    t = np.arange(len(price))
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, price, color='#2c3e50', alpha=0.6, lw=1.5, label='Asset Price')
    
    # Markers
    if model_res.liquidation_fraction is not None:
        liqs = model_res.liquidation_fraction[idx]
        
        # Partial: 0 < k < 1
        mask_partial = (liqs > 0) & (liqs < 1.0)
        if np.any(mask_partial):
            plt.scatter(t[mask_partial], price[mask_partial], 
                        c='#f1c40f', s=30, marker='o', label='Partial Liquidation', zorder=5)
        
        # Full: k = 1.0 or Default
        # Note: 'defaults' is boolean per path. We want TIME of default.
        # Usually k=1 implies default. Or look for last non-zero notional? 
        # Let's use k=1 as proxy for Full Closeout events.
        mask_full = (liqs >= 1.0)
        if np.any(mask_full):
            plt.scatter(t[mask_full], price[mask_full], 
                        c='#e74c3c', s=100, marker='x', linewidths=2.5, label='Full Closeout / Default', zorder=6)

    plt.title(f"Enhanced Liquidation Timeline (Path #{idx}) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Price ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_enhanced_liquidation.png"), dpi=150)
    plt.close()


def plot_leverage_landscape_heatmap(model_res: SingleModelResults, outdir: str):
    """
    Heatmap of System Leverage Distribution across Monte Carlo Paths over time.
    Since we aggregate traders per path, this shows 'Inter-Path' heterogeneity.
    
    X-Axis: Time
    Y-Axis: Leverage Buckets
    Color: Probability Density (% of paths in that bucket)
    """
    if model_res.lev_long is None:
        return
    
    # Combine Long and Short leverage? Or just max?
    # Let's use max(lev_long, lev_short) per path/tick to capture worst case
    lev_data = model_res.lev_long
    if model_res.lev_short is not None:
        lev_data = np.fmax(lev_data, model_res.lev_short)
    
    # Handle NaNs (dead paths) -> 0 leverage
    lev_data = np.nan_to_num(lev_data, nan=0.0)
    
    bins = [0, 5, 10, 20, 50, 100, 1000]
    bin_labels = ['0-5x', '5-10x', '10-20x', '20-50x', '50-100x', '>100x']
    
    P, T = lev_data.shape
    hist_grid = np.zeros((len(bins)-1, T))
    
    # Compute histogram per time step
    for t in range(T):
        counts, _ = np.histogram(lev_data[:, t], bins=bins)
        hist_grid[:, t] = counts / P # Normalize to probability
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(hist_grid, cmap="inferno", yticklabels=bin_labels, 
                cbar_kws={'label': 'Path Density (%)'}, vmin=0, vmax=1)
    
    plt.title(f"Leverage Landscape (System Max Lev across Paths) — {model_res.name}")
    plt.xlabel("Time Step")
    plt.ylabel("Leverage Bucket")
    plt.gca().invert_yaxis() # Low lev at bottom
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_leverage_landscape.png"), dpi=150)
    plt.close()


def plot_equity_at_risk_snapshot(model_res: SingleModelResults, outdir: str):
    """
    Distribution of End-of-Simulation System Equity across all paths.
    Approximates 'Equity at Risk' (EaR).
    
    Insight:
        - Tail risk of system insolvency.
        - Skewness of wealth outcomes.
    """
    if model_res.equity_long is None:
        return
    
    # Take final timestep (T-1)
    final_eq_long = model_res.equity_long[:, -1]
    final_eq_short = model_res.equity_short[:, -1]
    total_equity = final_eq_long + final_eq_short
    
    plt.figure(figsize=(10, 6))
    sns.histplot(total_equity, kde=True, stat='density', element='step', color='teal')
    
    # Mark Zero (Insolvency)
    plt.axvline(0, color='red', lw=2, ls='--', label='Insolvency Threshold ($0)')
    
    # Mark Mean
    mean_eq = np.mean(total_equity)
    plt.axvline(mean_eq, color='black', lw=1.5, ls=':', label=f'Mean Equity (${mean_eq:,.0f})')
    
    # Stats
    insolvency_prob = np.mean(total_equity < 0) * 100
    
    plt.title(f"End-State Equity Distribution (EaR) — {model_res.name}\nProb. of System Insolvency: {insolvency_prob:.2f}%")
    plt.xlabel("Total System Equity ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_equity_at_risk.png"), dpi=150)
    plt.close()


def plot_risk_diamond(model_res: SingleModelResults, outdir: str):
    """
    Scatterplot of Position Size vs Leverage, sized by Margin Usage.
    "The Risk Diamond": Identifies Systemic Whales vs Gamblers.
    
    Requires granular trader snapshots.
    """
    # Check for future 'trader_snapshots' attribute
    has_granular_data = hasattr(model_res, 'trader_snapshots') and model_res.trader_snapshots is not None
    
    plt.figure(figsize=(10, 8))
    
    if not has_granular_data:
        # Placeholder for missing data
        plt.text(0.5, 0.5, "Granular Trader Data Not Available\n(Engine update required to export snapshots)", 
                 ha='center', va='center', fontsize=14, color='gray',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
    else:
        # TODO: Implement scatter logic when data is available
        # x = positions, y = leverage, s = margin_usage
        pass

    plt.title(f"Risk Diamond (Trader Heterogeneity) — {model_res.name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_res.name}_risk_diamond.png"), dpi=150)
    plt.close()


# ==============================================================================
#  Drivers
# ==============================================================================


def plot_all_for_model(model_res: SingleModelResults, outdir: str, max_paths: int = 5):
    """Legacy driver for per-model plots, upgraded with new charts."""
    d = os.path.join(outdir, model_res.name)
    _ensure_dir(d)
    
    # Core
    plot_regime_dynamics(model_res, d)
    plot_liquidation_heatmap(model_res, d)
    plot_notional_fan_chart(model_res, d)
    plot_worst_case_autopsy(model_res, d)
    plot_slippage_waterfall(model_res, d)
    plot_system_leverage(model_res, d)
    plot_oi_with_breaker_bands(model_res, d)
    plot_systemic_solvency_curve(model_res, d)
    plot_trade_gating_waterfall(model_res, d)
    plot_enhanced_liquidation_timeline(model_res, d)
    plot_leverage_landscape_heatmap(model_res, d)
    plot_equity_at_risk_snapshot(model_res, d)
    plot_risk_diamond(model_res, d)


def plot_all(results: MultiModelResults, outdir: str):
    """
    Main entry point for the visualization suite.
    Generates all comparisons and per-model detailed charts.
    """
    _ensure_dir(outdir)
    
    print("Genering Comparison Dashboards...")
    plot_df_survival_curve(results, outdir)
    plot_model_comparison_violins(results, outdir)
    plot_efficiency_frontier(results, outdir)
    plot_mc_convergence(results, outdir)
    
    print("Generating Model Deep-Dives...")
    for name, model_res in results.models.items():
        plot_all_for_model(model_res, outdir)