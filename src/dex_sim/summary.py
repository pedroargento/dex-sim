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
    Option 2: percentiles computed on this global array.
    """
    parts = []
    if m.lev_long is not None:
        parts.append(m.lev_long.flatten())
    if m.lev_short is not None:
        parts.append(m.lev_short.flatten())

    if not parts:
        return np.array([], dtype=np.float64)

    arr = np.concatenate(parts)
    arr = arr[np.isfinite(arr)]
    return arr


def _lev_percentile(m: SingleModelResults, q: float) -> str:
    arr = _get_lev_array(m)
    if arr.size == 0:
        return _format_float(0.0)
    return _format_float(float(np.percentile(arr, q)))


# ------------------------------------------------------------
# Main summary generation
# ------------------------------------------------------------


def generate_summary(sim: MultiModelResults, out_dir: str) -> str:
    lines = []

    # ------------------------------------------------------------
    # 1. HEADER
    # ------------------------------------------------------------
    lines.append(f"# Simulation Summary Report: {out_dir.split('/')[-1]}\n")
    lines.append(f"**Timestamp:** {pd.Timestamp.now()}\n")
    lines.append(f"**Models:** {', '.join(sim.models.keys())}\n")

    # ------------------------------------------------------------
    # 2. SIMULATION PARAMETERS
    # ------------------------------------------------------------
    lines.append("## 1. Simulation Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Num Paths | {sim.num_paths:,} |")
    lines.append(f"| Horizon | {sim.horizon} steps |")
    lines.append(f"| Initial Price | {_format_currency(sim.initial_price)} |")
    lines.append(f"| Notional | {_format_currency(sim.notional)} |")

    stress = sim.metadata.get("stress_factor", "N/A")
    sigma = sim.metadata.get("sigma_daily", "N/A")
    lines.append(f"| Stress Factor | {stress} |")
    lines.append(f"| Sigma Daily | {sigma} |")
    lines.append("\n")

    # ------------------------------------------------------------
    # 3. MODEL COMPARISON SCORECARD
    # ------------------------------------------------------------
    lines.append("## 2. Model Comparison Scorecard\n")

    models = list(sim.models.values())
    model_names = [m.name for m in models]

    def get_metric_row(label, func):
        vals = [func(m) for m in models]
        return f"| {label} | " + " | ".join(str(v) for v in vals) + " |"

    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(
        get_metric_row(
            "Probability of Default (PoD)",
            lambda m: _format_pct(
                np.count_nonzero(m.df_required > 0) / len(m.df_required)
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Max Loss (Max DF)",
            lambda m: _format_currency(float(np.max(m.df_required))),
        )
    )

    lines.append(
        get_metric_row(
            "Tail Risk (VaR 99.9%)",
            lambda m: _format_currency(float(np.percentile(m.df_required, 99.9))),
        )
    )

    lines.append(
        get_metric_row(
            "Average Leverage (x)",
            lambda m: _format_float(
                float(np.mean(_get_lev_array(m))) if _get_lev_array(m).size > 0 else 0.0
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Avg Liquidation Size",
            lambda m: (
                "N/A"
                if m.liquidation_fraction is None
                else (
                    _format_currency(
                        float(
                            np.mean(m.liquidation_fraction[m.liquidation_fraction > 0])
                        )
                    )
                    if np.any(m.liquidation_fraction > 0)
                    else "$0.00"
                )
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Total System Slippage",
            lambda m: (
                _format_currency(float(np.sum(m.slippage_cost)))
                if m.slippage_cost is not None
                else "N/A"
            ),
        )
    )
    lines.append("\n")

    # ------------------------------------------------------------
    # 4. SOLVENCY & DF ANALYTICS
    # ------------------------------------------------------------
    lines.append("## 3. Solvency & Default Fund Analytics\n")

    # --- Default Fund Usage Statistics
    lines.append("### Default Fund Usage Statistics\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(
        get_metric_row(
            "Zero-Loss Paths (%)",
            lambda m: _format_pct(
                np.count_nonzero(m.df_required == 0) / len(m.df_required)
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Mean Loss (Conditional > 0)",
            lambda m: _format_currency(
                float(np.mean(m.df_required[m.df_required > 0]))
                if np.any(m.df_required > 0)
                else 0.0
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Median Loss (Conditional > 0)",
            lambda m: _format_currency(
                float(np.median(m.df_required[m.df_required > 0]))
                if np.any(m.df_required > 0)
                else 0.0
            ),
        )
    )

    lines.append(
        get_metric_row(
            "VaR 99.0%",
            lambda m: _format_currency(float(np.percentile(m.df_required, 99.0))),
        )
    )

    lines.append(
        get_metric_row(
            "Expected Shortfall (ES 99%)",
            lambda m: (
                _format_currency(
                    float(
                        np.mean(
                            m.df_required[
                                m.df_required > np.percentile(m.df_required, 99.0)
                            ]
                        )
                    )
                )
                if np.any(m.df_required > np.percentile(m.df_required, 99.0))
                else _format_currency(float(np.percentile(m.df_required, 99.0)))
            ),
        )
    )
    lines.append("\n")

    # --- Cost Composition
    lines.append("### Cost Composition\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def slippage_sum(m):
        return 0.0 if m.slippage_cost is None else float(np.sum(m.slippage_cost))

    def insolvency(m):
        total_df = float(np.sum(m.df_required))
        slip = slippage_sum(m)
        return max(0.0, total_df - slip)

    lines.append(
        get_metric_row(
            "Total Insolvency Loss", lambda m: _format_currency(insolvency(m))
        )
    )
    lines.append(
        get_metric_row(
            "Total Slippage Cost", lambda m: _format_currency(slippage_sum(m))
        )
    )
    lines.append(
        get_metric_row(
            "Insolvency Ratio",
            lambda m: (
                "0.00"
                if np.sum(m.df_required) == 0
                else f"{insolvency(m) / np.sum(m.df_required):.4f}"
            ),
        )
    )
    lines.append("\n")

    # ------------------------------------------------------------
    # 3.1 Slippage Breakdown (already present)
    # ------------------------------------------------------------
    lines.append("### 3.1 Slippage Breakdown (Trading vs Liquidation)\n")
    lines.append("| Component | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(get_metric_row("Trading Slippage", lambda m: _format_currency(0.0)))
    lines.append(
        get_metric_row(
            "Liquidation Slippage", lambda m: _format_currency(slippage_sum(m))
        )
    )
    lines.append(
        get_metric_row("Liquidation Slippage Share", lambda m: _format_pct(1.0))
    )
    lines.append("\n")

    # ------------------------------------------------------------
    # 3.2 NEW — Liquidation Size Diagnostics
    # ------------------------------------------------------------
    lines.append("### 3.2 Liquidation Size Diagnostics\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def total_liq_volume(m):
        if m.liquidation_fraction is None or m.notional_paths is None:
            return _format_currency(0.0)
        # liquidation_fraction[k] * notional → USD size
        liq_usd = m.liquidation_fraction * m.notional_paths
        return _format_currency(float(np.sum(liq_usd)))

    def mean_liq_usd(m):
        if m.liquidation_fraction is None or m.notional_paths is None:
            return _format_currency(0.0)
        liq_usd = m.liquidation_fraction * m.notional_paths
        mask = liq_usd > 0
        if not np.any(mask):
            return _format_currency(0.0)
        return _format_currency(float(np.mean(liq_usd[mask])))

    def p99_liq_usd(m):
        if m.liquidation_fraction is None or m.notional_paths is None:
            return _format_currency(0.0)
        liq_usd = (m.liquidation_fraction * m.notional_paths).flatten()
        liq_usd = liq_usd[liq_usd > 0]
        if liq_usd.size == 0:
            return _format_currency(0.0)
        return _format_currency(float(np.percentile(liq_usd, 99)))

    lines.append(get_metric_row("Total Liquidation USD Volume", total_liq_volume))
    lines.append(get_metric_row("Mean Liquidation USD", mean_liq_usd))
    lines.append(get_metric_row("P99 Liquidation USD", p99_liq_usd))
    lines.append("\n")

    # ------------------------------------------------------------
    # 3.3 NEW — Slippage Efficiency (slippage per USD liquidated)
    # ------------------------------------------------------------
    lines.append("### 3.3 Slippage Efficiency\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def slip_per_usd(m):
        if (
            m.liquidation_fraction is None
            or m.notional_paths is None
            or m.slippage_cost is None
        ):
            return "N/A"
        liq_usd = m.liquidation_fraction * m.notional_paths
        total_liq = float(np.sum(liq_usd))
        if total_liq == 0:
            return "N/A"
        total_slip = float(np.sum(m.slippage_cost))
        return _format_float(total_slip / total_liq)

    lines.append(get_metric_row("Slippage per $1 Liquidated", slip_per_usd))
    lines.append("\n")

    # ------------------------------------------------------------
    # 3.4 NEW — Liquidation Clustering Metrics
    # ------------------------------------------------------------
    lines.append("### 3.4 Liquidation Clustering\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    def liq_events_per_step(m):
        if m.liquidation_fraction is None:
            return _format_float(0.0)
        return _format_float(
            float(np.mean(np.count_nonzero(m.liquidation_fraction > 0, axis=0)))
        )

    def liq_event_variance(m):
        if m.liquidation_fraction is None:
            return _format_float(0.0)
        counts = np.count_nonzero(m.liquidation_fraction > 0, axis=0)
        return _format_float(float(np.var(counts)))

    def liq_peak_load(m):
        if m.liquidation_fraction is None:
            return _format_float(0.0)
        counts = np.count_nonzero(m.liquidation_fraction > 0, axis=0)
        return _format_float(float(np.max(counts)))

    lines.append(get_metric_row("Mean Liquidations / Step", liq_events_per_step))
    lines.append(get_metric_row("Variance of Liquidations / Step", liq_event_variance))
    lines.append(get_metric_row("Peak Liquidation Load", liq_peak_load))
    lines.append("\n")

    # ------------------------------------------------------------
    # 4. CAPITAL EFFICIENCY & LEVERAGE PROFILE
    # (unchanged below this point)
    # ------------------------------------------------------------

    lines.append("## 4. Capital Efficiency & Leverage Profile\n")
    lines.append("### Leverage Distribution\n")

    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(
        get_metric_row(
            "Mean System Leverage",
            lambda m: _format_float(
                float(np.mean(_get_lev_array(m))) if _get_lev_array(m).size > 0 else 0.0
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Median System Leverage",
            lambda m: _format_float(
                float(np.median(_get_lev_array(m)))
                if _get_lev_array(m).size > 0
                else 0.0
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Max Peak Leverage",
            lambda m: _format_float(
                float(np.max(_get_lev_array(m))) if _get_lev_array(m).size > 0 else 0.0
            ),
        )
    )

    lines.append(get_metric_row("Leverage P50", lambda m: _lev_percentile(m, 50)))
    lines.append(get_metric_row("Leverage P90", lambda m: _lev_percentile(m, 90)))
    lines.append(get_metric_row("Leverage P95", lambda m: _lev_percentile(m, 95)))
    lines.append(get_metric_row("Leverage P99", lambda m: _lev_percentile(m, 99)))

    lines.append(
        get_metric_row(
            "Time > 20x",
            lambda m: _format_pct(
                np.count_nonzero(_get_lev_array(m) > 20)
                / max(_get_lev_array(m).size, 1)
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Time > 50x",
            lambda m: _format_pct(
                np.count_nonzero(_get_lev_array(m) > 50)
                / max(_get_lev_array(m).size, 1)
            ),
        )
    )
    lines.append("\n")

    # ------------------------------------------------------------
    # 4.2 Position Survival
    # ------------------------------------------------------------
    lines.append("### Position Survival\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(
        get_metric_row(
            "Survival Rate",
            lambda m: (
                "N/A"
                if m.notional_paths is None
                else _format_pct(
                    np.count_nonzero(m.notional_paths[:, -1] > 1.0)
                    / m.notional_paths.shape[0]
                )
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Mean Final Notional",
            lambda m: (
                "N/A"
                if m.notional_paths is None
                else _format_currency(float(np.mean(m.notional_paths[:, -1])))
            ),
        )
    )
    lines.append("\n")

    # ------------------------------------------------------------
    # 5. Mechanism Dynamics (AES Only)
    # ------------------------------------------------------------
    lines.append("## 5. Mechanism Dynamics (AES Only)\n")

    aes_models = [
        m for m in models if m.breaker_state is not None and np.any(m.breaker_state)
    ]

    if aes_models:
        aes_names = [m.name for m in aes_models]

        lines.append("| Metric | " + " | ".join(aes_names) + " |")
        lines.append("| :--- | " + " | ".join([":---"] * len(aes_names)) + " |")

        def aes_row(label, fn):
            return f"| {label} | " + " | ".join(fn(m) for m in aes_models) + " |"

        lines.append(
            aes_row(
                "Normal State %",
                lambda m: _format_pct(
                    np.count_nonzero(m.breaker_state == 0) / m.breaker_state.size
                ),
            )
        )

        lines.append(
            aes_row(
                "Soft Breaker %",
                lambda m: _format_pct(
                    np.count_nonzero(m.breaker_state == 1) / m.breaker_state.size
                ),
            )
        )

        lines.append(
            aes_row(
                "Hard Breaker %",
                lambda m: _format_pct(
                    np.count_nonzero(m.breaker_state == 2) / m.breaker_state.size
                ),
            )
        )

        lines.append(
            aes_row(
                "Mean Margin Multiplier",
                lambda m: (
                    _format_float(float(np.mean(m.margin_multiplier)))
                    if m.margin_multiplier is not None
                    else "N/A"
                ),
            )
        )

        lines.append(
            aes_row(
                "Mean Rt",
                lambda m: (
                    _format_float(float(np.mean(m.rt))) if m.rt is not None else "N/A"
                ),
            )
        )

        lines.append(
            aes_row(
                "Max Rt",
                lambda m: (
                    _format_float(float(np.max(m.rt))) if m.rt is not None else "N/A"
                ),
            )
        )

        lines.append(
            aes_row(
                "Rt Volatility",
                lambda m: (
                    _format_float(float(np.std(m.rt))) if m.rt is not None else "N/A"
                ),
            )
        )

        lines.append("\n")

    else:
        lines.append(
            "*No models with active mechanism dynamics (breaker states) found.*\n"
        )

    # ------------------------------------------------------------
    # 6. LIQUIDATION MICROSTRUCTURE
    # ------------------------------------------------------------
    lines.append("## 6. Liquidation Microstructure\n")
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("| :--- | " + " | ".join([":---"] * len(models)) + " |")

    lines.append(
        get_metric_row(
            "Event Count",
            lambda m: (
                "0"
                if m.liquidation_fraction is None
                else f"{np.count_nonzero(m.liquidation_fraction > 0):,}"
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Mean Fraction (k)",
            lambda m: (
                "N/A"
                if m.liquidation_fraction is None
                else (
                    _format_float(
                        float(
                            np.mean(m.liquidation_fraction[m.liquidation_fraction > 0])
                        )
                    )
                    if np.any(m.liquidation_fraction > 0)
                    else "0.0000"
                )
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Full Closeouts",
            lambda m: (
                "0"
                if m.liquidation_fraction is None
                else f"{np.count_nonzero(m.liquidation_fraction == 1.0):,}"
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Partial Liquidations",
            lambda m: (
                "0"
                if m.liquidation_fraction is None
                else f"{np.count_nonzero((m.liquidation_fraction > 0) & (m.liquidation_fraction < 1.0)):,}"
            ),
        )
    )

    lines.append(
        get_metric_row(
            "Cascade Freq (>5% paths)",
            lambda m: (
                "0.00%"
                if m.liquidation_fraction is None
                else _format_pct(
                    np.count_nonzero(
                        np.count_nonzero(m.liquidation_fraction > 0, axis=0)
                        > 0.05 * m.liquidation_fraction.shape[0]
                    )
                    / m.liquidation_fraction.shape[1]
                )
            ),
        )
    )

    # ------------------------------------------------------------
    # Write output file
    # ------------------------------------------------------------
    filename = "summary.md"
    full_path = os.path.join(out_dir, filename)
    with open(full_path, "w") as f:
        f.write("\n".join(lines))

    return full_path
