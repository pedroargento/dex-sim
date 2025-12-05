#  dex-sim: DEX - CCP Risk Simulation

`dex-sim` is a high-performance risk simulation framework for modeling Central Counterparty (CCP) style margining, liquidations, and systemic stress for perpetual futures exchanges.

It is designed for speed, accuracy, and flexibility, allowing researchers and developers to test and compare different risk management models under extreme market conditions.

## Project Overview

`dex-sim` simulates the financial risk dynamics of a perpetual futures exchange operating under a CCP model. The core of the project is a powerful Monte Carlo engine that generates thousands of market scenarios to assess how different risk models perform.

- **Risk Modeling**: Supports fully modular component-based risk models (IM, Breaker, Liquidation).
- **High-Performance Engine**: Utilizes a **Numba-accelerated** Monte Carlo engine to simulate thousands of price paths with GARCH-based volatility models, ensuring both speed and realism.
- **Systemic Stress Testing**: Models complex dynamics like circuit breakers, dynamic margin adjustments, and default fund consumption.
- **Interactive Visualization**: Automatically generates Plotly dashboards (`dashboard.html`, `drilldown.html`) for deep exploration of risk metrics and individual simulation paths.

## Features

- **Pluggable Risk Models**: Build models by composing Initial Margin, Breaker, and Liquidation components.
- **Numba-Powered Monte Carlo**: Blazing fast 🔥 simulation core JIT-compiled with Numba.
- **Multi-Trader Engine**: Simulates a fixed pool of heterogeneous traders with individual margins, leverage, and behaviors.
- **Systemic Stress Index (Rₜ)**: A composite index that tracks real-time market stress by combining volatility, liquidity, and return shocks.
- **Circuit Breakers**: Implements a multi-state (NORMAL, SOFT, HARD) circuit breaker that dynamically adjusts margin multipliers in response to Rₜ.
- **Variation Margin (VM) Default Modeling**: Accurately simulates scenarios where a trader's equity is wiped out, triggering a default.
- **Default Fund Consumption**: Models the financial impact of defaults on the CCP's default fund, including slippage costs.
- **Efficient Result Storage**: Persists simulation outputs using the high-performance **Zarr** format.
- **Comprehensive CLI**:
    - `dex-sim run`: Run experiments from a YAML config.
    - `dex-sim list`: List all previous simulation runs.
    - `dex-sim plot`: Generate a full suite of plots for a given run.
    - `dex-sim compare`: Compare key metrics across multiple runs.

## Quickstart

### Installation

Requires Python 3.10+ and `uv` (or pip).

```bash
uv sync
# or
pip install .
```

### Running Your First Simulation

Run the included example configuration:

```bash
dex-sim run config/aes_vs_fxd_test.yaml
```

This command will:
1.  Generate 1,500 market paths using GARCH(1,1) volatility parameters.
2.  Simulate 2,000 traders (expanders/reducers) trading against a CCP.
3.  Compare two models: `AES_es99` (Dynamic Risk) vs `FXD_20x` (Fixed Leverage).
4.  Save results to `results/<timestamp>_aes_vs_fxd_test/`.
5.  Generate a **Summary Report** (`summary.md`) and **Dashboards** (`dashboard.html`, `drilldown.html`).

## Risk Model Architecture (Component-Based)

`dex-sim` uses a fully modular architecture. There is only one `RiskModel` class. A model is simply a composition of pluggable components defined in YAML:

1.  **Initial Margin (IM)**: Determines the base margin requirement.
    -   `type: es`: Expected Shortfall (Dynamic, volatility-based).
    -   `type: fixed_leverage`: Constant leverage (e.g., 20x).
2.  **Breaker**: A Finite State Machine that monitors systemic stress ($R_t$) and applies multipliers to the IM requirement.
    -   Defines `soft` and `hard` thresholds for $R_t$.
    -   Applies `multipliers` (e.g., `[1.0, 1.5, 2.0]`) corresponding to NORMAL, SOFT, and HARD states.
3.  **Liquidation**: Defines how positions are closed when margin is breached.
    -   `type: partial`: Closes only enough position to restore solvency (plus a buffer).
    -   `type: full`: Closes the entire position immediately.
    -   `slippage`: Configurable slippage factor for liquidation costs.

This architecture allows you to simulate almost any exchange design without writing new Python code.

## YAML Model Specification

Models are defined in the `models` list of your experiment configuration file.

```yaml
# Global Trader Pool Configuration
traders:
  count: 2000
  initial_equity: 50000.0
  behaviors:
    expand_fraction: 0.5  # Fraction of primary traders who expand positions
    expand_rate: 0.01     # Rate at which they add to positions
    reduce_fraction: 0.5  # Fraction of primary traders who reduce positions
    reduce_rate: 0.005    # Rate at which they close positions

models:
  - name: CCP_Numba_Accelerated
    backend: numba          # Enable high-performance engine
    im:
      type: es
      conf: 0.99
    breaker:
      soft: 1.0
      hard: 2.0
      multipliers: [1.0, 1.5, 2.0]
    liquidation:
      type: partial
      slippage: 0.001
```

**Defaults:**
*   `backend`: "python" (Reference implementation). Set to "numba" for speed.

### Example 1: CCP-Like Model (Cartesi CCP)
Uses dynamic Expected Shortfall margin, an active circuit breaker, and partial liquidation to mitigate cascades.

```yaml
models:
  - name: CCP_ES99
    backend: numba
    im:
      type: es
      conf: 0.99
    breaker:
      soft: 0.40
      hard: 0.70
      multipliers: [1.0, 1.1, 1.25]
    liquidation:
      type: partial
      slippage: 0.001
```

### Example 2: Hyperliquid-Style Fixed-Leverage Model
Simple constant leverage, no breaker, full liquidation on bankruptcy.

```yaml
models:
  - name: Hyperliquid_20x
    im:
      type: fixed_leverage
      leverage: 20
    breaker:
      soft: .inf
      hard: .inf
      multipliers: [1, 1, 1]
    liquidation:
      type: full
      slippage: 0.001
```

## Performance & Optimization

`dex-sim` includes two simulation backends:

1.  **Python (Reference)**: Flexible, object-oriented, useful for debugging and developing new logic.
2.  **Numba (Accelerated)**: JIT-compiled, array-based engine. Roughly **100x-500x faster**.

To enable the optimized engine, add `backend: numba` to your model config.

## 📑 Simulation Summary & Reporting

`dex-sim` includes a comprehensive reporting system that automatically generates a quantitative summary of the simulation results.

### Summary Report (`summary.md`)

This Markdown file provides a standardized "scorecard" for comparing model performance, including:

*   **Default Metrics**: Probability of Default (PoD), Max Loss, VaR 99.9%, Insolvency Loss.
*   **Leverage Profile**: Mean/Max System Leverage, Time spent > 20x/50x leverage.
*   **Liquidation Microstructure**: Total Event Count, Mean Fraction ($k$), Cascade Frequency (>5% paths/step).
*   **Slippage Efficiency**: Slippage cost per $1 liquidated.
*   **Systemic Stress (AES Only)**: Regime occupancy (% time in Soft/Hard breaker states), Mean Margin Multiplier.

### Interactive Dashboards

Two HTML dashboards are automatically generated:

1.  **`dashboard.html`**: A high-level comparison view with small-multiples for:
    -   **Risk Index**: Systemic risk ($R_t$) trends and breaker activation.
    -   **Leverage**: P50/P95 leverage bands across all paths.
    -   **Margin**: Initial Margin (IM) and Maintenance Margin (MM) bands.
    -   **Liquidation**: Intensity and cost of liquidations.
    -   **Trade Flow**: Accepted vs. Rejected trade volume.
    -   **Exposure**: Total Open Interest (OI) and Net ECP Exposure.

2.  **`drilldown.html`**: A forensic tool to explore individual simulation paths.
    -   Select any specific Model and Path ID.
    -   View synchronized plots of Price, $R_t$, Leverage, and Equity.
    -   Inspect individual Liquidation events (marked with ❌).

## Results Storage

Simulation results are stored in the `results/<run_id>/` directory using **Zarr** for efficient array storage.

**Directory Structure:**
```
results/20251205_123456_experiment/
├── data.zarr/              # Zarr root group
│   ├── log_returns         # MC Input: Log returns [P, T]
│   ├── sigma_path          # MC Input: GARCH volatility [P, T]
│   └── models/
│       └── AES_es99/       # Per-model outputs
│           ├── price_paths # [P, T]
│           ├── lev_long    # [P, T]
│           ├── rt          # [P, T] (Risk Index)
│           ├── breaker_state
│           ├── liquidation_fraction
│           └── ...
├── metadata.json           # Global simulation parameters
├── summary.md              # Text report
├── dashboard.html          # Interactive plots
└── drilldown.html          # Path explorer
```

## Extending Models

You can add new components by creating a Python class in `src/dex_sim/models/components.py` and registering it in `experiment_manager.py`.

1.  **New IM Strategy**: Subclass `InitialMargin`. Implement `compute(notional, sigma)`.
2.  **New Liquidation Logic**: Subclass `LiquidationStrategy`.
3.  **Register**: Update the `build_im`, `build_breaker`, or `build_liquidation` functions in `src/dex_sim/experiment_manager.py` to handle your new `type` string.

## Development Setup

The project follows a standard Python project structure.

```
dex-sim/
├── config/                 # Experiment configuration files
├── results/                # Simulation outputs
├── src/
│   └── dex_sim/
│       ├── cli.py          # Command-line interface
│       ├── engine.py       # Orchestrator
│       ├── engine_numba_columnar.py # Optimized kernel
│       ├── experiment_manager.py # Config parser & runner
│       ├── models/         # Risk model definitions
│       ├── plotting/       # Dashboard generation
│       │   ├── dashboard.py
│       │   ├── dashboard_export.py
│       │   └── panels/     # Visualization components
│       ├── results_io.py   # Zarr I/O
│       └── summary.py      # Metric calculation
├── pyproject.toml          # Dependencies
└── README.md
```