import os
import numpy as np
import pandas as pd
from .data_structures import MultiModelResults, SingleModelResults


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------


def _format_currency(val: float) -> str:
    return f"${val:,.2f}"


def _format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _format_float(val: float) -> str:
    return f"{val:.4f}"


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size > 0 else 0.0


def _safe_percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size > 0 else 0.0


# ------------------------------------------------------------
# Leverage helpers
# ------------------------------------------------------------


def _get_lev_array(m: SingleModelResults) -> np.ndarray:
    """
    Flatten all leverage values (long + short) into a single 1D array.

    Option 2: percentiles are computed on this global array across
    *all* paths and *all* timesteps, with no engine changes.

    We also drop non-finite values to avoid NaN propagation.
    """
    parts = []
    if m.lev_long is not None:
        parts.append(m.lev_long.flatten())
    if m.lev_short is not None:
        parts.append(m.lev_short.flatten())

    if not parts:
        return np.array([], dtype=np.float64)

    arr = np.concatenate(parts)
    # Drop NaN / inf just in case
    arr = arr[np.isfinite(arr)]
    return arr


def _lev_percentile(m: SingleModelResults, q: float) -> str:
    """
    Percentile of leverage using the flattened global array.
    Returns formatted 0.0000 if array is empty.
    """
    arr = _get_lev_array(m)
    if arr.size == 0:
        return _format_float(0.0)
    val = np.percentile(arr, q)
    return _format_float(float(val))


# ------------------------------------------------------------
# Main summary generation
# ------------------------------------------------------------


def generate_summary(sim: MultiModelResults, out_dir: str) -> str:
    """
    Generates a purely quantitative markdown summary report based on the simulation results.
    Follows strict 'no narrative' rule.
    """
    lines = []

    # --- Header ---
    lines.append(f"# Simulation Summary Report: {out_dir.split('/')[-1]}\n")
    lines.append(f"**Timestamp:** {pd.Timestamp.now()}\n")
    lines.append(f"**Models:** {', '.join(sim.models.keys())}\n")

    # --- 1. Simulation Parameters ---
    lines.append("## 1. Simulation Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Num Paths | {sim.num_paths:,} |")
    lines.append(f"| Horizon | {sim.horizon} steps |")
    lines.append(f"| Initial Price | {_format_currency(sim.initial_price)} |")
    lines.append(f"| Notional | {_format_currency(sim.notional)} |")

    # Metadata
    stress = sim.metadata.get("stress_factor", "N/A")
    sigma = sim.metadata.get("sigma_daily", "N/A")
    lines.append(f"| Stress Factor | {stress} |")
    lines.append(f"| Sigma Daily | {sigma} |")
    lines.append("\n")

    # --- 2. Model Comparison Scorecard ---
    lines.append("## 2. Model Comparison Scorecard\n")

    models = list(sim.models.values())
    model_names = [m.name for m in models]

    def get_metric_row(metric_name: str, func) -> str:
        vals = [func(m) for m in models]
        return f"| {metric_name} | " + " | ".join(str(v) for v in vals) + " |"

    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    # 2.1 Probability of Default (PoD)
    def calc_pod(m: SingleModelResults) -> str:
        return _format_pct(np.count_nonzero(m.df_required > 0) / len(m.df_required))

    lines.append(get_metric_row("Probability of Default (PoD)", calc_pod))

    # 2.2 Max Loss (Max DF)
    def calc_max_loss(m: SingleModelResults) -> str:
        return _format_currency(float(np.max(m.df_required)))

    lines.append(get_metric_row("Max Loss (Max DF)", calc_max_loss))

    # 2.3 Tail Risk (VaR 99.9%)
    def calc_var999(m: SingleModelResults) -> str:
        return _format_currency(float(np.percentile(m.df_required, 99.9)))

    lines.append(get_metric_row("Tail Risk (VaR 99.9%)", calc_var999))

    # 2.4 Average Leverage (x) – global flatten
    def calc_avg_lev(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_float(0.0)
        return _format_float(float(np.mean(arr)))

    lines.append(get_metric_row("Average Leverage (x)", calc_avg_lev))

    # 2.5 Average Liquidation Size (conditional)
    def calc_avg_liq_size(m: SingleModelResults) -> str:
        # partial_liq_amount is optional / model-dependent
        if not hasattr(m, "partial_liq_amount") or m.partial_liq_amount is None:
            return "N/A"
        mask = m.partial_liq_amount > 0
        if not np.any(mask):
            return "$0.00"
        return _format_currency(float(np.mean(m.partial_liq_amount[mask])))

    lines.append(get_metric_row("Avg Liquidation Size", calc_avg_liq_size))

    # 2.6 Total System Slippage
    def calc_total_slippage(m: SingleModelResults) -> str:
        if m.slippage_cost is None:
            return "N/A"
        return _format_currency(float(np.sum(m.slippage_cost)))

    lines.append(get_metric_row("Total System Slippage", calc_total_slippage))
    lines.append("\n")

    # --- 3. Solvency & Default Fund Analytics ---
    lines.append("## 3. Solvency & Default Fund Analytics\n")

    # 3.1 Default Fund Usage Statistics
    lines.append("### Default Fund Usage Statistics\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def calc_zero_loss_pct(m: SingleModelResults) -> str:
        return _format_pct(np.count_nonzero(m.df_required == 0) / len(m.df_required))

    lines.append(get_metric_row("Zero-Loss Paths (%)", calc_zero_loss_pct))

    def calc_mean_loss_cond(m: SingleModelResults) -> str:
        mask = m.df_required > 0
        val = float(np.mean(m.df_required[mask])) if np.any(mask) else 0.0
        return _format_currency(val)

    lines.append(get_metric_row("Mean Loss (Conditional > 0)", calc_mean_loss_cond))

    def calc_median_loss_cond(m: SingleModelResults) -> str:
        mask = m.df_required > 0
        val = float(np.median(m.df_required[mask])) if np.any(mask) else 0.0
        return _format_currency(val)

    lines.append(get_metric_row("Median Loss (Conditional > 0)", calc_median_loss_cond))

    def calc_var99(m: SingleModelResults) -> str:
        return _format_currency(float(np.percentile(m.df_required, 99.0)))

    lines.append(get_metric_row("VaR 99.0%", calc_var99))

    def calc_es99(m: SingleModelResults) -> str:
        var99 = float(np.percentile(m.df_required, 99.0))
        mask = m.df_required > var99
        val = float(np.mean(m.df_required[mask])) if np.any(mask) else var99
        return _format_currency(val)

    lines.append(get_metric_row("Expected Shortfall (ES 99%)", calc_es99))
    lines.append("\n")

    # 3.2 Cost Composition
    lines.append("### Cost Composition\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def calc_slippage_sum(m: SingleModelResults) -> float:
        if m.slippage_cost is None:
            return 0.0
        return float(np.sum(m.slippage_cost))

    def calc_insolvency_loss(m: SingleModelResults) -> float:
        total_df = float(np.sum(m.df_required))
        slip = calc_slippage_sum(m)
        # Approximation: DF usage includes both bankruptcy hole + slippage.
        # Insolvency is what remains after slippage.
        return max(0.0, total_df - slip)

    def fmt_insolvency(m: SingleModelResults) -> str:
        return _format_currency(calc_insolvency_loss(m))

    lines.append(get_metric_row("Total Insolvency Loss", fmt_insolvency))

    def fmt_slippage(m: SingleModelResults) -> str:
        return _format_currency(calc_slippage_sum(m))

    lines.append(get_metric_row("Total Slippage Cost", fmt_slippage))

    def calc_ratio(m: SingleModelResults) -> str:
        insolv = calc_insolvency_loss(m)
        total = float(np.sum(m.df_required))
        if total == 0.0:
            return "0.00"
        return f"{insolv / total:.4f}"

    lines.append(get_metric_row("Insolvency Ratio", calc_ratio))
    lines.append("\n")

    # 3.3 Slippage Breakdown (Trading vs Liquidation)
    # For now, the engine only accounts slippage during liquidations,
    # so "Trading Slippage" is zero and all slippage is attributed to liquidation.
    lines.append("### 3.1 Slippage Breakdown (Trading vs Liquidation)\n")
    lines.append("| Component | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def trading_slip_fmt(m: SingleModelResults) -> str:
        # No trading slippage modelled yet
        return _format_currency(0.0)

    def liquidation_slip_fmt(m: SingleModelResults) -> str:
        return fmt_slippage(m)

    def liquidation_share_fmt(m: SingleModelResults) -> str:
        total = calc_slippage_sum(m)
        if total == 0.0:
            return _format_pct(0.0)
        # Currently 100% by construction
        return _format_pct(1.0)

    lines.append(get_metric_row("Trading Slippage", trading_slip_fmt))
    lines.append(get_metric_row("Liquidation Slippage", liquidation_slip_fmt))
    lines.append(get_metric_row("Liquidation Slippage Share", liquidation_share_fmt))
    lines.append("\n")

    # --- 4. Capital Efficiency & Leverage Profile ---
    lines.append("## 4. Capital Efficiency & Leverage Profile\n")

    # 4.1 Leverage Distribution
    lines.append("### Leverage Distribution\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def calc_mean_sys_lev(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_float(0.0)
        return _format_float(float(np.mean(arr)))

    lines.append(get_metric_row("Mean System Leverage", calc_mean_sys_lev))

    def calc_median_sys_lev(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_float(0.0)
        return _format_float(float(np.median(arr)))

    lines.append(get_metric_row("Median System Leverage", calc_median_sys_lev))

    def calc_max_peak_lev(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_float(0.0)
        return _format_float(float(np.max(arr)))

    lines.append(get_metric_row("Max Peak Leverage", calc_max_peak_lev))

    # New: Leverage percentiles using Option 2 (global flatten)
    lines.append(get_metric_row("Leverage P50", lambda m: _lev_percentile(m, 50.0)))
    lines.append(get_metric_row("Leverage P90", lambda m: _lev_percentile(m, 90.0)))
    lines.append(get_metric_row("Leverage P95", lambda m: _lev_percentile(m, 95.0)))
    lines.append(get_metric_row("Leverage P99", lambda m: _lev_percentile(m, 99.0)))

    def calc_time_above_20(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_pct(0.0)
        count = np.count_nonzero(arr > 20.0)
        total = arr.size
        return _format_pct(count / total)

    lines.append(get_metric_row("Time > 20x", calc_time_above_20))

    def calc_time_above_50(m: SingleModelResults) -> str:
        arr = _get_lev_array(m)
        if arr.size == 0:
            return _format_pct(0.0)
        count = np.count_nonzero(arr > 50.0)
        total = arr.size
        return _format_pct(count / total)

    lines.append(get_metric_row("Time > 50x", calc_time_above_50))
    lines.append("\n")

    # 4.2 Position Survival
    lines.append("### Position Survival\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def calc_survival_rate(m: SingleModelResults) -> str:
        if m.notional_paths is None:
            return "N/A"
        final_notional = m.notional_paths[:, -1]
        survived = np.count_nonzero(final_notional > 1.0)  # small tolerance
        return _format_pct(survived / len(final_notional))

    lines.append(get_metric_row("Survival Rate", calc_survival_rate))

    def calc_mean_final_notional(m: SingleModelResults) -> str:
        if m.notional_paths is None:
            return "N/A"
        return _format_currency(float(np.mean(m.notional_paths[:, -1])))

    lines.append(get_metric_row("Mean Final Notional", calc_mean_final_notional))
    lines.append("\n")

    # --- 5. Mechanism Dynamics (AES Only) ---
    lines.append("## 5. Mechanism Dynamics (AES Only)\n")

    aes_models = [
        m for m in models if m.breaker_state is not None and np.any(m.breaker_state)
    ]
    if not aes_models:
        lines.append(
            "*No models with active mechanism dynamics (breaker states) found.*\n"
        )
    else:
        aes_names = [m.name for m in aes_models]
        lines.append("| Metric | " + " | ".join(aes_names) + " |")
        lines.append("| :--- | " + " | ".join([":---"] * len(aes_models)) + " |")

        def get_aes_row(metric_name: str, func) -> str:
            vals = [func(m) for m in aes_models]
            return f"| {metric_name} | " + " | ".join(str(v) for v in vals) + " |"

        def calc_normal_pct(m: SingleModelResults) -> str:
            return _format_pct(
                np.count_nonzero(m.breaker_state == 0) / m.breaker_state.size
            )

        lines.append(get_aes_row("Normal State %", calc_normal_pct))

        def calc_soft_pct(m: SingleModelResults) -> str:
            return _format_pct(
                np.count_nonzero(m.breaker_state == 1) / m.breaker_state.size
            )

        lines.append(get_aes_row("Soft Breaker %", calc_soft_pct))

        def calc_hard_pct(m: SingleModelResults) -> str:
            return _format_pct(
                np.count_nonzero(m.breaker_state == 2) / m.breaker_state.size
            )

        lines.append(get_aes_row("Hard Breaker %", calc_hard_pct))

        def calc_mean_mult(m: SingleModelResults) -> str:
            if m.margin_multiplier is None:
                return "N/A"
            return _format_float(float(np.mean(m.margin_multiplier)))

        lines.append(get_aes_row("Mean Margin Multiplier", calc_mean_mult))

        def calc_mean_rt(m: SingleModelResults) -> str:
            if m.rt is None:
                return "N/A"
            return _format_float(float(np.mean(m.rt)))

        lines.append(get_aes_row("Mean Rt", calc_mean_rt))

        def calc_max_rt(m: SingleModelResults) -> str:
            if m.rt is None:
                return "N/A"
            return _format_float(float(np.max(m.rt)))

        lines.append(get_aes_row("Max Rt", calc_max_rt))

        def calc_vol_rt(m: SingleModelResults) -> str:
            if m.rt is None:
                return "N/A"
            return _format_float(float(np.std(m.rt)))

        lines.append(get_aes_row("Rt Volatility", calc_vol_rt))
        lines.append("\n")

    # --- 6. Liquidation Microstructure ---
    lines.append("## 6. Liquidation Microstructure\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def calc_event_count(m: SingleModelResults) -> str:
        if m.liquidation_fraction is None:
            return "0"
        return f"{np.count_nonzero(m.liquidation_fraction > 0):,}"

    lines.append(get_metric_row("Event Count", calc_event_count))

    def calc_mean_k(m: SingleModelResults) -> str:
        if m.liquidation_fraction is None:
            return "N/A"
        mask = m.liquidation_fraction > 0
        if not np.any(mask):
            return "0.0000"
        return _format_float(float(np.mean(m.liquidation_fraction[mask])))

    lines.append(get_metric_row("Mean Fraction (k)", calc_mean_k))

    def calc_full_closeouts(m: SingleModelResults) -> str:
        if m.liquidation_fraction is None:
            return "0"
        return f"{np.count_nonzero(m.liquidation_fraction == 1.0):,}"

    lines.append(get_metric_row("Full Closeouts", calc_full_closeouts))

    def calc_partial_liqs(m: SingleModelResults) -> str:
        if m.liquidation_fraction is None:
            return "0"
        mask = (m.liquidation_fraction > 0) & (m.liquidation_fraction < 1.0)
        return f"{np.count_nonzero(mask):,}"

    lines.append(get_metric_row("Partial Liquidations", calc_partial_liqs))

    def calc_cascade_freq(m: SingleModelResults) -> str:
        if m.liquidation_fraction is None:
            return "0.00%"
        liqs_per_t = np.count_nonzero(m.liquidation_fraction > 0, axis=0)
        threshold = 0.05 * m.liquidation_fraction.shape[0]
        cascade_steps = np.count_nonzero(liqs_per_t > threshold)
        return _format_pct(cascade_steps / m.liquidation_fraction.shape[1])

    lines.append(get_metric_row("Cascade Freq (>5% paths)", calc_cascade_freq))

    # --- Write to file ---
    filename = "summary.md"
    full_path = os.path.join(out_dir, filename)
    with open(full_path, "w") as f:
        f.write("\n".join(lines))

    return full_path
