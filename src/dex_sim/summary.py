import os
import numpy as np
import pandas as pd
from .data_structures import MultiModelResults, SingleModelResults

def _format_currency(val):
    return f"${val:,.2f}"

def _format_pct(val):
    return f"{val * 100:.2f}%"

def _format_float(val):
    return f"{val:.4f}"

def _safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def _safe_median(arr):
    return np.median(arr) if len(arr) > 0 else 0.0

def _safe_percentile(arr, p):
    return np.percentile(arr, p) if len(arr) > 0 else 0.0

def generate_summary(sim: MultiModelResults, out_dir: str):
    """
    Generates a purely quantitative markdown summary report based on the simulation results.
    Follows strict 'no narrative' rule.
    """
    
    lines = []
    
    # --- Header ---
    lines.append(f"# Simulation Summary Report: {out_dir.split('/')[-1]}\n")
    lines.append(f"**Timestamp:** {pd.Timestamp.now()}\n")
    lines.append(f"**Models:** {', '.join(sim.models.keys())}\n")
    
    # --- Simulation Parameters ---
    lines.append("## 1. Simulation Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Num Paths | {sim.num_paths:,} |")
    lines.append(f"| Horizon | {sim.horizon} steps |")
    lines.append(f"| Initial Price | {_format_currency(sim.initial_price)} |")
    lines.append(f"| Notional | {_format_currency(sim.notional)} |")
    # Extract metadata safely
    stress = sim.metadata.get("stress_factor", "N/A")
    sigma = sim.metadata.get("sigma_daily", "N/A")
    lines.append(f"| Stress Factor | {stress} |")
    lines.append(f"| Sigma Daily | {sigma} |")
    lines.append("\n")

    # --- I. Model Comparison Scorecard ---
    lines.append("## 2. Model Comparison Scorecard\n")
    
    # Prepare data for comparison
    models = list(sim.models.values())
    model_names = [m.name for m in models]
    
    # Helper to get metric list for all models
    def get_metric_row(metric_name, func):
        vals = [func(m) for m in models]
        row = f"| {metric_name} | " + " | ".join(str(v) for v in vals) + " |"
        return row

    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    # 1. Probability of Default (PoD)
    def calc_pod(m):
        return _format_pct(np.count_nonzero(m.df_required > 0) / len(m.df_required))
    lines.append(get_metric_row("Probability of Default (PoD)", calc_pod))
    
    # 2. Max Loss
    def calc_max_loss(m):
        return _format_currency(np.max(m.df_required))
    lines.append(get_metric_row("Max Loss (Max DF)", calc_max_loss))
    
    # 3. Tail Risk (VaR 99.9%)
    def calc_var999(m):
        return _format_currency(np.percentile(m.df_required, 99.9))
    lines.append(get_metric_row("Tail Risk (VaR 99.9%)", calc_var999))
    
    # 4. Average Leverage
    def calc_avg_lev(m):
        # Combine long and short leverage, ignoring NaNs
        lev_all = np.concatenate([m.lev_long.flatten(), m.lev_short.flatten()])
        return _format_float(np.nanmean(lev_all))
    lines.append(get_metric_row("Average Leverage (x)", calc_avg_lev))
    
    # 5. Average Liquidation Size (conditional)
    def calc_avg_liq_size(m):
        if m.partial_liq_amount is None: return "N/A"
        mask = m.partial_liq_amount > 0
        if not np.any(mask): return "$0.00"
        return _format_currency(np.mean(m.partial_liq_amount[mask]))
    lines.append(get_metric_row("Avg Liquidation Size", calc_avg_liq_size))
    
    # 6. Total System Slippage
    def calc_total_slippage(m):
        if m.slippage_cost is None: return "N/A"
        return _format_currency(np.sum(m.slippage_cost))
    lines.append(get_metric_row("Total System Slippage", calc_total_slippage))
    lines.append("\n")

    # --- II. Solvency & Default Fund Analytics ---
    lines.append("## 3. Solvency & Default Fund Analytics\n")
    
    # Table 2: Default Fund Usage Statistics
    lines.append("### Default Fund Usage Statistics\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    def calc_zero_loss_pct(m):
        return _format_pct(np.count_nonzero(m.df_required == 0) / len(m.df_required))
    lines.append(get_metric_row("Zero-Loss Paths (%)", calc_zero_loss_pct))
    
    def calc_mean_loss_cond(m):
        mask = m.df_required > 0
        val = np.mean(m.df_required[mask]) if np.any(mask) else 0.0
        return _format_currency(val)
    lines.append(get_metric_row("Mean Loss (Conditional > 0)", calc_mean_loss_cond))
    
    def calc_median_loss_cond(m):
        mask = m.df_required > 0
        val = np.median(m.df_required[mask]) if np.any(mask) else 0.0
        return _format_currency(val)
    lines.append(get_metric_row("Median Loss (Conditional > 0)", calc_median_loss_cond))
    
    def calc_var99(m):
        return _format_currency(np.percentile(m.df_required, 99.0))
    lines.append(get_metric_row("VaR 99.0%", calc_var99))
    
    def calc_es99(m):
        var99 = np.percentile(m.df_required, 99.0)
        mask = m.df_required > var99
        val = np.mean(m.df_required[mask]) if np.any(mask) else var99
        return _format_currency(val)
    lines.append(get_metric_row("Expected Shortfall (ES 99%)", calc_es99))
    lines.append("\n")
    
    # Table 3: Cost Composition
    lines.append("### Cost Composition\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    def calc_slippage_sum(m):
        if m.slippage_cost is None: return 0.0
        return np.sum(m.slippage_cost)
    
    def calc_insolvency_loss(m):
        total_df = np.sum(m.df_required)
        slip = calc_slippage_sum(m)
        # Insolvency = DF - Slippage (Rough approx if specific breakdown not available per path)
        # Engine adds slippage to DF usage. 
        # But DF usage also includes bankruptcy hole.
        # So Insolvency = Total DF - Total Slippage.
        val = max(0.0, total_df - slip)
        return val

    def fmt_insolvency(m): return _format_currency(calc_insolvency_loss(m))
    lines.append(get_metric_row("Total Insolvency Loss", fmt_insolvency))
    
    def fmt_slippage(m): return _format_currency(calc_slippage_sum(m))
    lines.append(get_metric_row("Total Slippage Cost", fmt_slippage))
    
    def calc_ratio(m):
        insolv = calc_insolvency_loss(m)
        total = np.sum(m.df_required)
        if total == 0: return "0.00"
        return f"{insolv / total:.4f}"
    lines.append(get_metric_row("Insolvency Ratio", calc_ratio))
    lines.append("\n")

    # --- III. Capital Efficiency & Leverage Profile ---
    lines.append("## 4. Capital Efficiency & Leverage Profile\n")
    
    lines.append("### Leverage Distribution\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    def _get_lev_array(m):
        return np.concatenate([m.lev_long.flatten(), m.lev_short.flatten()])
    
    def calc_mean_sys_lev(m):
        return _format_float(np.nanmean(_get_lev_array(m)))
    lines.append(get_metric_row("Mean System Leverage", calc_mean_sys_lev))
    
    def calc_median_sys_lev(m):
        return _format_float(np.nanmedian(_get_lev_array(m)))
    lines.append(get_metric_row("Median System Leverage", calc_median_sys_lev))
    
    def calc_max_peak_lev(m):
        return _format_float(np.nanmax(_get_lev_array(m)))
    lines.append(get_metric_row("Max Peak Leverage", calc_max_peak_lev))
    
    def calc_time_above_20(m):
        arr = _get_lev_array(m)
        count = np.count_nonzero(arr > 20)
        total = arr.size
        return _format_pct(count/total)
    lines.append(get_metric_row("Time > 20x", calc_time_above_20))
    
    def calc_time_above_50(m):
        arr = _get_lev_array(m)
        count = np.count_nonzero(arr > 50)
        total = arr.size
        return _format_pct(count/total)
    lines.append(get_metric_row("Time > 50x", calc_time_above_50))
    lines.append("\n")
    
    lines.append("### Position Survival\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    def calc_survival_rate(m):
        # Check final timestep notional > 0
        if m.notional_paths is None: return "N/A"
        final_notional = m.notional_paths[:, -1]
        survived = np.count_nonzero(final_notional > 1.0) # tolerance
        return _format_pct(survived / len(final_notional))
    lines.append(get_metric_row("Survival Rate", calc_survival_rate))
    
    def calc_mean_final_notional(m):
        if m.notional_paths is None: return "N/A"
        return _format_currency(np.mean(m.notional_paths[:, -1]))
    lines.append(get_metric_row("Mean Final Notional", calc_mean_final_notional))
    lines.append("\n")

    # --- IV. Mechanism Dynamics (AES Only) ---
    lines.append("## 5. Mechanism Dynamics (AES Only)\n")
    
    # Filter for AES models (those with breaker_state)
    aes_models = [m for m in models if m.breaker_state is not None and np.any(m.breaker_state)]
    
    if not aes_models:
        lines.append("*No models with active mechanism dynamics (breaker states) found.*\n")
    else:
        aes_names = [m.name for m in aes_models]
        lines.append("| Metric | " + " | ".join(aes_names) + " |")
        lines.append("| :--- | " + " | ".join([":---"] * len(aes_models)) + " |")
        
        # Helper specifically for AES subset
        def get_aes_row(metric_name, func):
            vals = [func(m) for m in aes_models]
            return f"| {metric_name} | " + " | ".join(str(v) for v in vals) + " |"

        def calc_normal_pct(m):
            return _format_pct(np.count_nonzero(m.breaker_state == 0) / m.breaker_state.size)
        lines.append(get_aes_row("Normal State %", calc_normal_pct))
        
        def calc_soft_pct(m):
            return _format_pct(np.count_nonzero(m.breaker_state == 1) / m.breaker_state.size)
        lines.append(get_aes_row("Soft Breaker %", calc_soft_pct))
        
        def calc_hard_pct(m):
            return _format_pct(np.count_nonzero(m.breaker_state == 2) / m.breaker_state.size)
        lines.append(get_aes_row("Hard Breaker %", calc_hard_pct))
        
        def calc_mean_mult(m):
            if m.margin_multiplier is None: return "N/A"
            return _format_float(np.mean(m.margin_multiplier))
        lines.append(get_aes_row("Mean Margin Multiplier", calc_mean_mult))
        
        def calc_mean_rt(m):
            if m.rt is None: return "N/A"
            return _format_float(np.mean(m.rt))
        lines.append(get_aes_row("Mean Rt", calc_mean_rt))
        
        def calc_max_rt(m):
            if m.rt is None: return "N/A"
            return _format_float(np.max(m.rt))
        lines.append(get_aes_row("Max Rt", calc_max_rt))
        
        def calc_vol_rt(m):
            if m.rt is None: return "N/A"
            return _format_float(np.std(m.rt))
        lines.append(get_aes_row("Rt Volatility", calc_vol_rt))
        lines.append("\n")

    # --- V. Liquidation Microstructure ---
    lines.append("## 6. Liquidation Microstructure\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")
    
    def calc_event_count(m):
        if m.liquidation_fraction is None: return "0"
        return f"{np.count_nonzero(m.liquidation_fraction > 0):,}"
    lines.append(get_metric_row("Event Count", calc_event_count))
    
    def calc_mean_k(m):
        if m.liquidation_fraction is None: return "N/A"
        mask = m.liquidation_fraction > 0
        if not np.any(mask): return "0.0000"
        return _format_float(np.mean(m.liquidation_fraction[mask]))
    lines.append(get_metric_row("Mean Fraction (k)", calc_mean_k))
    
    def calc_full_closeouts(m):
        if m.liquidation_fraction is None: return "0"
        return f"{np.count_nonzero(m.liquidation_fraction == 1.0):,}"
    lines.append(get_metric_row("Full Closeouts", calc_full_closeouts))
    
    def calc_partial_liqs(m):
        if m.liquidation_fraction is None: return "0"
        mask = (m.liquidation_fraction > 0) & (m.liquidation_fraction < 1.0)
        return f"{np.count_nonzero(mask):,}"
    lines.append(get_metric_row("Partial Liquidations", calc_partial_liqs))
    
    def calc_cascade_freq(m):
        if m.liquidation_fraction is None: return "0.00%"
        # Count per timestep: how many paths liquidated?
        # shape: [P, T]
        liqs_per_t = np.count_nonzero(m.liquidation_fraction > 0, axis=0)
        # Threshold: 5% of paths
        threshold = 0.05 * m.liquidation_fraction.shape[0]
        cascade_steps = np.count_nonzero(liqs_per_t > threshold)
        return _format_pct(cascade_steps / m.liquidation_fraction.shape[1])
    lines.append(get_metric_row("Cascade Freq (>5% paths)", calc_cascade_freq))
    
    # Write to file
    filename = "summary.md"
    full_path = os.path.join(out_dir, filename)
    
    with open(full_path, "w") as f:
        f.write("\n".join(lines))
    
    return full_path
