"""
plotting.py — Professional Visualization Suite for Dex-Sim Risk Engine.

This module provides a comprehensive set of visualizations to analyze:
1. System-Level Risk Dashboard (Time-series, Leverage, Liquidation)
2. Microstructure Explorer (Individual trader behavior, equity maps)
3. Symmetry Diagnostics (Net exposure, ECP absorption)
4. Model Comparison (DF distribution, Leverage profiles)

Designed for use with Matplotlib and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from scipy.stats import t as scipy_t  # avoid conflict with time t

from .data_structures import MultiModelResults, SingleModelResults
from .models.components import ES_IM, FixedLeverageIM, Breaker


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
    if np.all(model_res.df_required == 0):
        # If no defaults, pick path with highest liquidation fraction sum
        if model_res.liquidation_fraction is not None:
            return int(np.argmax(np.sum(model_res.liquidation_fraction, axis=1)))
        return 0
    return int(np.argmax(model_res.df_required))


def _save_fig(fig, outdir, filename):
    """Saves figure to PNG and PDF."""
    fig.savefig(os.path.join(outdir, f"{filename}.png"), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(outdir, f"{filename}.pdf"), bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
#  1. System-Level Risk Dashboard
# ==============================================================================

def plot_system_dashboard(model_res: SingleModelResults, outdir: str):
    """
    A single composite figure generating:
    A. Time-Series Panel: OI, ECP exposure, DF cumulative loss, Margin multipliers, Breaker states
    B. Leverage Panel: Mean, median, 5–95 percentile
    C. Liquidation Panel: Liquidation frequency heatmap, Mean fraction k, Slippage vs time
    """
    idx = _get_worst_path_idx(model_res)
    T = model_res.price_paths.shape[1]
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1.5, 1, 1, 1.2])

    # --- Row 0: Time-Series Panel (OI, ECP, DF, Breakers) ---
    
    # 0,0: Open Interest & ECP
    ax_oi = fig.add_subplot(gs[0, 0])
    if model_res.notional_paths is not None:
        oi = model_res.notional_paths[idx]
        ax_oi.fill_between(t_axis, 0, oi, color='tab:blue', alpha=0.3, label='System OI')
    
    if model_res.ecp_position_path is not None:
        # ECP position is in units. Convert to Notional approx.
        ecp_notional = np.abs(model_res.ecp_position_path[idx] * model_res.price_paths[idx])
        ax_oi.plot(t_axis, ecp_notional, color='tab:red', lw=1.5, label='ECP Exposure ($)')
    
    ax_oi.set_title("Open Interest & ECP Exposure")
    ax_oi.set_ylabel("Notional ($)")
    ax_oi.legend(loc='upper left')
    
    # 0,1: Cumulative DF Loss & Slippage
    ax_df = fig.add_subplot(gs[0, 1])
    if hasattr(model_res, 'df_path') and model_res.df_path is not None:
         cum_df = np.cumsum(model_res.df_path[idx])
         ax_df.plot(t_axis, cum_df, color='black', lw=2, label='Cumulative DF Loss')
    
    if model_res.slippage_cost is not None:
        cum_slip = np.cumsum(model_res.slippage_cost[idx])
        ax_df.plot(t_axis, cum_slip, color='gray', ls='--', label='Cumulative Slippage')
        
    ax_df.set_title("Cumulative Losses")
    ax_df.set_ylabel("Value ($)")
    ax_df.legend()

    # 0,2: Margin Multipliers & Breaker States
    ax_brk = fig.add_subplot(gs[0, 2])
    if model_res.margin_multiplier is not None:
        ax_brk.plot(t_axis, model_res.margin_multiplier[idx], color='purple', lw=2, label='Margin Mult')
    
    if model_res.breaker_state is not None:
        bs = model_res.breaker_state[idx]
        # Shade background
        # 0=Normal (Green), 1=Soft (Orange), 2=Hard (Red)
        # Use 3-arg form: X_range, Y_range, Data(1, T)
        ax_brk.pcolorfast([0, T], [0, 1], [bs], cmap='RdYlGn_r', alpha=0.2, vmin=0, vmax=2)
        
    ax_brk.set_title("Risk Regime")
    ax_brk.set_ylabel("Multiplier")
    
    # --- Row 1: Leverage Panel ---
    
    # 1,0: System Leverage Distribution (Fan Chart)
    ax_lev = fig.add_subplot(gs[1, :])
    if model_res.lev_long is not None and model_res.lev_short is not None:
        # Max leverage across Long/Short
        lev = np.fmax(model_res.lev_long, model_res.lev_short)
        # Clean NaNs
        lev = np.nan_to_num(lev, nan=0.0)
        
        p05 = np.percentile(lev, 5, axis=0)
        p50 = np.percentile(lev, 50, axis=0)
        p95 = np.percentile(lev, 95, axis=0)
        
        ax_lev.plot(t_axis, p50, color='navy', lw=1.5, label='Median Leverage')
        ax_lev.fill_between(t_axis, p05, p95, color='blue', alpha=0.1, label='5-95% Range')
        ax_lev.set_title("System Leverage Distribution (Across All Paths)")
        ax_lev.set_ylabel("Leverage (x)")
        ax_lev.legend()

    # --- Row 2: Liquidation Panel ---
    
    # 2,0: Liquidation Frequency Heatmap
    ax_liq_map = fig.add_subplot(gs[2, 0])
    if model_res.liquidation_fraction is not None:
        # Sort paths by total liquidation volume or DF
        sorted_idx = np.argsort(np.sum(model_res.liquidation_fraction, axis=1))
        # Downsample paths for heatmap if too many
        if len(sorted_idx) > 500:
            sorted_idx = sorted_idx[::int(len(sorted_idx)/500)]
            
        liq_data = model_res.liquidation_fraction[sorted_idx]
        sns.heatmap(liq_data, ax=ax_liq_map, cmap="Reds", cbar=False, xticklabels=False, yticklabels=False)
        ax_liq_map.set_title("Liquidation Intensity (Paths vs Time)")
        ax_liq_map.set_ylabel("Paths (Sorted)")

    # 2,1: Mean Liquidation Fraction (k)
    ax_k = fig.add_subplot(gs[2, 1])
    if model_res.liquidation_fraction is not None:
        # Mean non-zero k
        # Mask zeros
        k_masked = np.where(model_res.liquidation_fraction > 0, model_res.liquidation_fraction, np.nan)
        mean_k = np.nanmean(k_masked, axis=0)
        ax_k.plot(t_axis, mean_k, color='crimson', lw=1)
        ax_k.set_title("Avg Liquidation Size (k) when Active")
        ax_k.set_ylabel("Fraction k")

    # 2,2: Cascade Intensity
    ax_casc = fig.add_subplot(gs[2, 2])
    if model_res.liquidation_fraction is not None:
        # Count % of paths with liq > 0 at each tick
        liq_count = np.sum(model_res.liquidation_fraction > 0, axis=0)
        liq_freq = liq_count / model_res.df_required.shape[0]
        ax_casc.plot(t_axis, liq_freq, color='darkorange')
        ax_casc.axhline(0.05, color='red', ls='--', label='Cascade Threshold (5%)')
        ax_casc.set_title("Cascade Intensity (% Paths Liquidating)")
        ax_casc.legend()

    # --- Row 3: Equity Health Map (Microstructure Proxy) ---
    ax_eq_dist = fig.add_subplot(gs[3, :])
    if model_res.equity_long is not None:
        final_eq = (model_res.equity_long[:, -1] + model_res.equity_short[:, -1])
        # Handle potential NaN/Inf in histplot
        final_eq = final_eq[np.isfinite(final_eq)]
        sns.histplot(final_eq, ax=ax_eq_dist, bins=100, kde=True, color='teal')
        ax_eq_dist.set_title("Final System Equity Distribution")
        ax_eq_dist.axvline(0, color='red', lw=2, label='Insolvency')
        ax_eq_dist.set_xlabel("Equity ($)")

    plt.tight_layout()
    _save_fig(fig, outdir, f"{model_res.name}_system_dashboard")


# ==============================================================================
#  2. Microstructure Explorer
# ==============================================================================

def plot_microstructure_explorer(model_res: SingleModelResults, outdir: str):
    """
    Detailed view of individual trader behavior if data is available.
    """
    # Placeholder for future extension or if data is added.
    if hasattr(model_res, 'trader_snapshots') and model_res.trader_snapshots:
        _plot_risk_diamond(model_res, outdir)


def _plot_risk_diamond(model_res: SingleModelResults, outdir: str):
    """
    Scatterplot: Position Size vs Leverage, sized by Margin Usage.
    """
    snaps = model_res.trader_snapshots
    if not snaps:
        return

    positions = np.array([s['position'] for s in snaps])
    leverages = np.array([s['leverage'] for s in snaps])
    mm_usage = np.array([s['mm_usage'] for s in snaps])
    equities = np.array([s['equity'] for s in snaps])
    
    plt.figure(figsize=(10, 8))
    abs_pos = np.maximum(np.abs(positions), 1.0) 
    sizes = mm_usage * 200 + 20
    
    scatter = plt.scatter(abs_pos, leverages, s=sizes, c=equities, 
                          cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
    
    plt.xscale('log')
    plt.xlabel("Position Size ($) [Log Scale]")
    plt.ylabel("Effective Leverage (x)")
    plt.colorbar(scatter, label='Trader Equity ($)')
    plt.axhline(20, color='red', ls=':', alpha=0.5)
    
    plt.title(f"Risk Diamond (Trader Microstructure) — {model_res.name}")
    plt.tight_layout()
    _save_fig(plt.gcf(), outdir, f"{model_res.name}_risk_diamond")


# ==============================================================================
#  3. Symmetry Diagnostics
# ==============================================================================

def plot_symmetry_diagnostics(model_res: SingleModelResults, outdir: str):
    """
    Verifies market neutrality invariants.
    """
    idx = _get_worst_path_idx(model_res)
    T = model_res.price_paths.shape[1]
    t_axis = np.arange(T)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # A. Net Exposure vs ECP
    ax0 = axes[0]
    if model_res.ecp_position_path is not None:
        ecp_pos = model_res.ecp_position_path[idx]
        ax0.plot(t_axis, ecp_pos, color='purple', label='ECP Net Position')
        ax0.axhline(0, color='black', ls='--')
        ax0.set_ylabel("Units")
        ax0.set_title(f"ECP Absorption (Symmetry Breaker) — Path #{idx}")
        ax0.legend()
    
    # B. ECP Slippage / Cost
    ax1 = axes[1]
    if model_res.slippage_cost is not None:
        cum_slip = np.cumsum(model_res.slippage_cost[idx])
        ax1.plot(t_axis, cum_slip, color='gray', label='Cumulative System Slippage')
        ax1.set_ylabel("Cost ($)")
        ax1.set_title("Slippage Accumulation")
        ax1.legend()

    plt.tight_layout()
    _save_fig(fig, outdir, f"{model_res.name}_symmetry_diagnostics")


# ==============================================================================
#  4. Model Comparison Plots
# ==============================================================================

def plot_comparison_dashboard(results: MultiModelResults, outdir: str):
    """
    Side-by-side comparison of key metrics across all models.
    """
    # A. Default Fund Distribution (Box Plot)
    data_df = []
    for name, m in results.models.items():
        losses = m.df_required[m.df_required > 0]
        if len(losses) > 0:
            df_temp = pd.DataFrame({"Model": name, "DF Usage": losses})
            data_df.append(df_temp)
            
    if data_df:
        df_all = pd.concat(data_df)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_all, x="Model", y="DF Usage", palette="muted")
        plt.yscale('log')
        plt.title("Default Fund Loss Distribution (Log Scale)")
        plt.tight_layout()
        _save_fig(plt.gcf(), outdir, "comparison_df_boxplot")

    # B. Max Leverage Comparison (Violin)
    data_lev = []
    for name, m in results.models.items():
        if m.lev_long is not None:
            # Peak leverage per path
            peak_lev = np.max(m.lev_long, axis=1)
            # Check NaNs
            peak_lev = peak_lev[np.isfinite(peak_lev)]
            df_temp = pd.DataFrame({"Model": name, "Peak Leverage": peak_lev})
            data_lev.append(df_temp)
            
    if data_lev:
        df_lev_all = pd.concat(data_lev)
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_lev_all, x="Model", y="Peak Leverage", cut=0, scale="width")
        plt.title("Peak Leverage Distribution")
        plt.tight_layout()
        _save_fig(plt.gcf(), outdir, "comparison_leverage_violin")

    # C. OI Growth Comparison
    plt.figure(figsize=(10, 6))
    for name, m in results.models.items():
        if m.notional_paths is not None:
            # Median OI path
            median_oi = np.median(m.notional_paths, axis=0)
            plt.plot(median_oi, label=name, lw=2)
    plt.title("Median Open Interest Growth")
    plt.xlabel("Time")
    plt.ylabel("Notional ($)")
    plt.legend()
    plt.tight_layout()
    _save_fig(plt.gcf(), outdir, "comparison_oi_growth")


def plot_margin_dashboard(results: MultiModelResults, outdir: str):
    """
    Generates a 3-panel vertical dashboard analyzing margin mechanics across all models.
    
    Panel 1: IM/MM Requirement per $1 Notional (Median + 10-90% band)
    Panel 2: Volatility (σ) Distribution (Median + 10-90% band)
    Panel 3: Systemic Stress (R_t) with Breaker Regime Shading
    
    Args:
        results: MultiModelResults object containing data for all models.
        outdir: Directory to save the plot.
    """
    # 1. Setup
    target_dir = os.path.join(outdir, "margin_dashboard")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Shared inputs
    sigma = results.sigma_path # (P, T)
    if sigma is None:
        print("Warning: No sigma_path found in results. Skipping margin dashboard.")
        return

    T = sigma.shape[1]
    t_axis = np.arange(T)
    
    # Create Figure
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    
    # Consistent colors for models
    model_names = list(results.models.keys())
    colors = sns.color_palette("husl", len(model_names))
    model_colors = dict(zip(model_names, colors))

    # ------------------------------------------------------------
    # Panel 1: IM/MM Requirements per $1 Notional
    # ------------------------------------------------------------
    ax0 = axes[0]
    
    for name, model_res in results.models.items():
        meta = model_res.metadata
        es_factor = 0.0
        is_es = False
        
        # Heuristic to determine IM type
        im_type = meta.get('im_type', 'unknown')
        if 'es' in im_type.lower() or 'AES' in name:
            conf = float(meta.get('im_conf', 0.99))
            im_obj = ES_IM(conf=conf, df=6)
            es_factor = im_obj.compute(1.0, 1.0)
            is_es = True
        elif 'fixed' in im_type.lower() or 'FXD' in name:
            lev = float(meta.get('im_leverage', 10.0)) # Default to 10 if missing
            es_factor = 1.0 / lev
            is_es = False
        else:
            es_factor = 0.1
            is_es = False

        margin_mult = model_res.margin_multiplier
        if margin_mult is None:
            margin_mult = np.ones_like(sigma)
            
        if is_es:
            im_path = sigma * es_factor * margin_mult
        else:
            im_path = np.full_like(sigma, es_factor) * margin_mult
            
        mm_path = im_path * 0.8
        
        im_med = np.median(im_path, axis=0)
        im_p10 = np.percentile(im_path, 10, axis=0)
        im_p90 = np.percentile(im_path, 90, axis=0)
        mm_med = np.median(mm_path, axis=0)
        
        c = model_colors[name]
        
        ax0.fill_between(t_axis, im_p10, im_p90, color=c, alpha=0.15)
        ax0.plot(t_axis, im_med, color=c, label=f"{name} IM")
        ax0.plot(t_axis, mm_med, color=c, linestyle="--", alpha=0.7)

    ax0.set_title("IM/MM Requirement per $1 Notional")
    ax0.set_ylabel("Rate")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Panel 2: Volatility (σ) Distribution
    # ------------------------------------------------------------
    ax1 = axes[1]
    
    sigma_med = np.median(sigma, axis=0)
    sigma_p10 = np.percentile(sigma, 10, axis=0)
    sigma_p90 = np.percentile(sigma, 90, axis=0)
    
    ax1.fill_between(t_axis, sigma_p10, sigma_p90, color="gray", alpha=0.2, label="10-90% Range")
    ax1.plot(t_axis, sigma_med, color="black", lw=1.5, label="Median σ")
    
    ax1.set_title("Volatility (σ) Distribution")
    ax1.set_ylabel("Daily Volatility")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Panel 3: Systemic Stress R_t + Breaker Regimes
    # ------------------------------------------------------------
    ax2 = axes[2]
    
    for name, model_res in results.models.items():
        if model_res.rt is None or model_res.breaker_state is None:
            continue
            
        rt = model_res.rt
        breaker_state = model_res.breaker_state
        
        rt_med = np.median(rt, axis=0)
        c = model_colors[name]
        ax2.plot(t_axis, rt_med, color=c, label=f"{name} Median $R_t$")
        
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 0, breaker_state)
        mode_state = np.argmax(counts, axis=0)
        
        changes = np.where(np.diff(mode_state) != 0)[0] + 1
        segments = np.split(t_axis, changes)
        state_vals = np.split(mode_state, changes)
        
        for seg, st in zip(segments, state_vals):
            if len(seg) == 0: continue
            start, end = seg[0], seg[-1]
            s_val = st[0]
            
            if s_val == 0:
                col = 'green'; alpha = 0.05
            elif s_val == 1:
                col = 'orange'; alpha = 0.10
            else:
                col = 'red'; alpha = 0.15
                
            ax2.axvspan(start, end, color=col, alpha=alpha, lw=0)

    ax2.set_title("Systemic Stress $R_t$ with Breaker Regimes (Mode State)")
    ax2.set_ylabel("$R_t$")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, target_dir, "margin_dashboard")


# ==============================================================================
#  Legacy / Individual Plots (Re-implemented/Kept for compatibility)
# ==============================================================================

def plot_df_survival_curve(results: MultiModelResults, outdir: str):
    """Log-Log Survival Function (Reverse CDF) of Default Fund Usage."""
    plt.figure(figsize=(10, 6))
    has_data = False
    for name, model_res in results.models.items():
        data = np.sort(model_res.df_required)
        data = data[data > 0]
        if len(data) == 0:
            continue
        has_data = True
        n = len(data)
        y = np.arange(n, 0, -1) / len(model_res.df_required)
        plt.plot(data, y, lw=2.5, label=name)

    if has_data:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("DF Usage Amount ($) [Log]")
        plt.ylabel("Probability P(Loss > x) [Log]")
        plt.title("Solvency Survival Curve (Tail Risk Analysis)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        _save_fig(plt.gcf(), outdir, "solvency_survival_curve")
    else:
        plt.close()

def plot_efficiency_frontier(results: MultiModelResults, outdir: str):
    """Scatter plot of Safety (Max DF Usage) vs Efficiency (Average Margin)."""
    plt.figure(figsize=(10, 6))
    for name, model_res in results.models.items():
        if model_res.margin_multiplier is not None:
            avg_mult = np.mean(model_res.margin_multiplier)
        else:
            avg_mult = 1.0
        # Use 99.9% DF Usage
        if len(model_res.df_required) > 0:
            p99_loss = np.percentile(model_res.df_required, 99.9)
            plt.scatter(avg_mult, p99_loss, s=100, label=name, alpha=0.8, edgecolors='w')
            plt.text(avg_mult, p99_loss, f"  {name}", fontsize=9, va='center')

    plt.xlabel("Average Margin Multiplier (Capital Inefficiency)")
    plt.ylabel("99.9% DF Requirement ($)")
    plt.title("Safety vs. Efficiency Frontier")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    _save_fig(plt.gcf(), outdir, "efficiency_frontier")

def plot_liquidation_heatmap(model_res: SingleModelResults, outdir: str):
    """Legacy wrapper for heatmap."""
    # Already implemented inside plot_system_dashboard, but kept as individual file
    if model_res.liquidation_fraction is None: return
    sorted_idx = np.argsort(np.sum(model_res.liquidation_fraction, axis=1))
    if len(sorted_idx) > 500:
        sorted_idx = sorted_idx[::int(len(sorted_idx)/500)]
    liq_data = model_res.liquidation_fraction[sorted_idx]
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(liq_data, cmap="Reds", cbar=False, xticklabels=500, yticklabels=False)
    plt.title(f"Liquidation Intensity Heatmap — {model_res.name}")
    plt.tight_layout()
    _save_fig(plt.gcf(), outdir, f"{model_res.name}_liquidation_heatmap")

def plot_worst_case_autopsy(model_res: SingleModelResults, outdir: str):
    """Legacy wrapper for autopsy."""
    # Similar to system dashboard panel but more detailed
    idx = _get_worst_path_idx(model_res)
    T = model_res.price_paths.shape[1]
    t = np.arange(T)
    
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax0.plot(t, model_res.price_paths[idx], color='black', label='Price')
    if model_res.liquidation_fraction is not None:
        liqs = model_res.liquidation_fraction[idx]
        mask = liqs > 0
        ax0.scatter(t[mask], model_res.price_paths[idx][mask], color='red', s=10, label='Liquidation')
    ax0.set_title(f"Worst Case Autopsy (Path #{idx}) — {model_res.name}")
    ax0.legend()
    
    if model_res.equity_long is not None:
        ax1.plot(t, model_res.equity_long[idx], label='Long Equity')
        ax1.plot(t, model_res.equity_short[idx], label='Short Equity')
        ax1.axhline(0, color='red', ls='--')
        ax1.legend()
        
    plt.tight_layout()
    _save_fig(fig, outdir, f"{model_res.name}_autopsy")


# ==============================================================================
#  Main Entry Point
# ==============================================================================

def plot_all(results: MultiModelResults, outdir: str):
    """
    Main driver for generating all visualizations.
    """
    _ensure_dir(outdir)
    print(f"Generating plots in {outdir}...")

    # 1. Model-Specific Plots
    for name, model_res in results.models.items():
        mdir = os.path.join(outdir, name)
        _ensure_dir(mdir)
        
        print(f"  - Plotting dashboard for {name}...")
        plot_system_dashboard(model_res, mdir)
        plot_microstructure_explorer(model_res, mdir)
        plot_symmetry_diagnostics(model_res, mdir)
        
        # Legacy individual plots
        plot_liquidation_heatmap(model_res, mdir)
        plot_worst_case_autopsy(model_res, mdir)

    # 2. Comparison Plots
    if len(results.models) > 1:
        print("  - Plotting comparisons...")
        plot_comparison_dashboard(results, outdir)
        plot_df_survival_curve(results, outdir)
        plot_efficiency_frontier(results, outdir)
        plot_margin_dashboard(results, outdir)
    
    print("Done.")