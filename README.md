#  dex-sim: DEX - CCP Risk Simulation

`dex-sim` is a high-performance risk simulation framework for modeling Central Counterparty (CCP) style margining, liquidations, and systemic stress for perpetual futures exchanges.

It is designed for speed, accuracy, and flexibility, allowing researchers and developers to test and compare different risk management models under extreme market conditions.

## Project Overview

`dex-sim` simulates the financial risk dynamics of a perpetual futures exchange operating under a CCP model. The core of the project is a powerful Monte Carlo engine that generates thousands of market scenarios to assess how different risk models perform.

- **Risk Modeling**: Supports fully modular component-based risk models (IM, Breaker, Liquidation).
- **High-Performance Engine**: Utilizes a **Numba-accelerated** Monte Carlo engine to simulate thousands of price paths with GARCH-based volatility models, ensuring both speed and realism.
- **Systemic Stress Testing**: Models complex dynamics like circuit breakers, dynamic margin adjustments, and default fund consumption.

## Features

- **Pluggable Risk Models**: Build models by composing Initial Margin, Breaker, and Liquidation components.
- **Numba-Powered Monte Carlo**: Blazing fast 🔥 simulation core JIT-compiled with Numba.
- **Multi-Trader Engine**: Simulates heterogeneous trader populations with individual margins, leverage, and lifecycle events.
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

## Risk Model Architecture (Component-Based)

`dex-sim` uses a fully modular architecture. There is only one `RiskModel` class. A model is simply a composition of pluggable components defined in YAML:

1.  **Initial Margin (IM)**: Determines the base margin requirement (e.g., `ES_IM` for dynamic risk, `FixedLeverageIM` for constant leverage).
2.  **Breaker**: A Finite State Machine that monitors systemic stress ($R_t$) and applies multipliers to the IM requirement.
3.  **Liquidation**: Defines how positions are closed when margin is breached (e.g., `FullCloseOut`, `PartialCloseOut`).
4.  **Trader Arrival**: Configures the population dynamics (entry rate, leverage distribution, equity distribution).

This architecture allows you to simulate almost any exchange design without writing new Python code.

## YAML Model Specification

Models are defined in the `models` list of your experiment configuration file.

```yaml
# Global Trader Configuration
trader_arrival:
  enabled: true
  pairs_per_tick: 2
  equity_distribution: fixed
  equity_dist_params: {value: 10000}
  leverage_range: [2, 20]

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
*   `trader_arrival`: Disabled by default. Can be set globally or per-model.

### Example 1: CCP-Like Model (Cartesi CCP)
Uses dynamic Expected Shortfall margin, an active circuit breaker, and partial liquidation to mitigate cascades.

```yaml
trader_arrival:
  enabled: true
  pairs_per_tick: 5

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

### Example 3: Synthetic CFD Model
Fixed leverage baseline but with a system-wide breaker to hike margins during volatility, plus partial liquidation.

```yaml
models:
  - name: Synthetic_CFD
    im:
      type: fixed_leverage
      leverage: 10
    breaker:
      soft: 0.5
      hard: 0.9
      multipliers: [1.0, 1.2, 1.4]
    liquidation:
      type: partial
      slippage: 0.002
```

### Example 4: GMX-Style GLP Model (Delta-Neutral Liquidity)
Simulates a liquidity pool model where traders have effectively infinite initial leverage (no IM check at open), and liquidation is purely based on equity depletion. No breaker logic.

```yaml
models:
  - name: GMX_GLP
    im:
      type: fixed_leverage
      leverage: 1000000     # Effectively no IM check
    breaker:
      soft: .inf
      hard: .inf
      multipliers: [1, 1, 1]
    liquidation:
      type: full
      slippage: 0.001
```

### Example 5: DYDX-v4-Style Model
Orderbook-like leverage with a mild breaker regime.

```yaml
models:
  - name: DYDX_v4
    im:
      type: es
      conf: 0.95
    breaker:
      soft: 0.8
      hard: .inf
      multipliers: [1.0, 1.05, 1.05]
    liquidation:
      type: full
      slippage: 0.0005
```

## Performance & Optimization

`dex-sim` includes two simulation backends:

1.  **Python (Reference)**: Flexible, object-oriented, useful for debugging and developing new logic.
2.  **Numba (Accelerated)**: JIT-compiled, array-based engine. roughly **100x-500x faster**.

To enable the optimized engine, add `backend: numba` to your model config.

```yaml
models:
  - name: High_Performance_Run
    backend: numba
    # ... other config ...
```

## Extending Models

You can add new components by creating a Python class and registering it in `experiment_manager.py`.

1.  **New IM Strategy**: Subclass `InitialMargin` in `models/components.py`. Implement `compute(notional, sigma)`.
2.  **New Liquidation Logic**: Subclass `LiquidationStrategy`.
3.  **Register**: Update the `build_im`, `build_breaker`, or `build_liquidation` functions in `src/dex_sim/experiment_manager.py` to handle your new `type` string.

No subclassing of `RiskModel` is required.

## Full End-to-End Example Config

```yaml
name: multi_risk_test
paths: 10000
initial_price: 1800
notional: 400000000
stress_factor: 1.0
seed: 123
garch_params: garch_params.json

models:
  - name: CCP
    backend: numba
    im: { type: es, conf: 0.99 }
    breaker: { soft: 0.4, hard: 0.7, multipliers: [1.0, 1.1, 1.25] }
    liquidation: { type: partial }
    trader_arrival: { enabled: true, pairs_per_tick: 2 }

  - name: HL_20x
    backend: numba
    im: { type: fixed_leverage, leverage: 20 }
    breaker: { soft: .inf, hard: .inf, multipliers: [1,1,1] }
    liquidation: { type: full }
```

## 📑 Simulation Summary & Reporting

`dex-sim` includes a comprehensive reporting system that automatically generates a quantitative summary of the simulation results. This summary is designed to provide risk committees, researchers, and developers with immediate, actionable metrics derived directly from the simulation data.

### Purpose
- Aggregates all numerical risk metrics derived from `SimulationResults`.
- Provides a standardized "scorecard" for comparing model performance.
- Saved automatically to: `results/<run_id>/summary.md`.

### Manual Generation
You can manually regenerate the summary for any existing run:

```python
from dex_sim.results_io import load_results
from dex_sim.summary import generate_summary

# Load results and generate summary
results = load_results("results/20231129_my_experiment")
generate_summary(results, "results/20231129_my_experiment")
```

### Documented Metrics

The summary report includes the following data-derived metrics, grouped by category:

#### 1) Default & Waterfall Metrics
*   **Probability of Default (PoD)**: Percentage of paths where the Default Fund was utilized (`count(df > 0) / total_paths`).
*   **DF Usage Distribution**: Mean, Median, 99th percentile (VaR), and Max DF usage across all paths.
*   **Expected Shortfall (ES 99%)**: Average DF usage for the worst 1% of outcomes.
*   **Insolvency Loss**: Total loss due to negative equity (bankruptcy).

#### 2) Leverage Metrics
*   **Mean System Leverage**: Average leverage across all paths and timesteps.
*   **Max Peak Leverage**: The highest single leverage value observed in any path/timestep.
*   **Time > 20x/50x**: Percentage of simulation time where leverage exceeded 20x or 50x.

#### 3) Systemic Stress Metrics (Rₜ)
*   **Mean/Max Rₜ**: Average and peak values of the systemic risk index.
*   **Rₜ Volatility**: Standard deviation of the risk index.

#### 4) Breaker Metrics
*   **Regime Occupancy**: Percentage of time spent in `NORMAL`, `SOFT`, and `HARD` breaker states.
*   **Mean Margin Multiplier**: The average multiplier applied to Initial Margin requirements.

#### 5) Liquidation Metrics
*   **Event Count**: Total number of liquidation events across all paths.
*   **Mean Fraction ($k$)**: Average portion of the position closed during a liquidation event.
*   **Cascade Frequency**: Percentage of timesteps where >5% of paths liquidated simultaneously.
*   **Full vs. Partial**: Counts of full closeouts ($k=1.0$) vs. partial reductions ($k < 1.0$).

#### 6) Slippage Metrics
*   **Total Slippage Cost**: Sum of all slippage costs incurred by the system.
*   **Cost Composition**: Breakdown of total losses into Insolvency (Gap Risk) vs. Slippage (Liquidity Risk).

#### 7) Exposure & Notional Metrics
*   **Survival Rate**: Percentage of paths that ended with a non-zero position.
*   **Mean Final Notional**: Average remaining open interest at the end of the simulation.

#### 8) Model Comparison Tables
*   **Scorecard**: A side-by-side comparison table of key metrics (PoD, Max Loss, VaR, Avg Leverage) for all models in the experiment.

## 📊 Plotting & Visualization

`dex-sim` includes a professional-grade visualization suite (`src/dex_sim/plotting.py`) powered by `matplotlib` and `seaborn`. This system is designed to move beyond simple averages and reveal the **tail risks**, **liquidation cascades**, and **systemic dynamics** hidden within the Monte Carlo data.

Plots are automatically generated after every `dex-sim run`. You can also generate them manually for any past run using the CLI or Python API.

### Output Location
All charts are saved as high-resolution PNGs in:
```
results/<run_id>/plots/
```

### Manual Usage (Python)
```python
from dex_sim.results_io import load_results
from dex_sim.plotting import plot_all

# Load simulation data
results = load_results("results/20231129_my_experiment")

# Generate full suite
plot_all(results, "results/20231129_my_experiment/plots")
```

---

### 📉 Documentation of Charts

Below is a guide to the charts produced by the suite and how to interpret them from a risk perspective.

#### 1. Solvency Survival Curve (Log-Log) 💀
**What it is:** A reverse cumulative distribution (survival function) of Default Fund (DF) usage.
**Interpretation:**
*   **The Curve:** Shows the probability (Y-axis) that a loss will exceed a certain amount (X-axis).
*   **The Tail:** A straight line on this log-log plot indicates "heavy tails" (power-law distribution), meaning catastrophic losses are more likely than a standard bell curve predicts.

#### 2. Systemic Solvency Curve (Health Monitor) 🛡️
**What it is:** Dual-line chart comparing Total Trader Equity (Green) vs Total Maintenance Margin (Red).
**Interpretation:**
*   **Green Zone:** Solvency Buffer. The system is healthy.
*   **Red Zone:** Insolvency/Fragility. Equity < MM means the system is undercollateralized and relying on the insurance fund.

#### 3. OI Timeline with Breaker Bands 🛑
**What it is:** Stacked Area Chart of Open Interest overlaid with vertical bands for Breaker States (Green=Normal, Orange=Soft, Red=Hard).
**Interpretation:**
*   **Efficacy:** Did the breaker trigger *before* the crash?
*   **Gating:** Did OI contract (or stop growing) when the background turned Red?

#### 4. Trade Intent Gating Waterfall 🚪
**What it is:** Stacked Bar Chart showing volume Accepted (Normal), Accepted (Risk-Reducing), and Rejected.
**Interpretation:**
*   **Gating:** Visualizes the "Reduce-Only" mechanism in action.
*   **Red Bars:** Volume physically blocked by the risk engine.

#### 5. Leverage Landscape Heatmap ⚖️
**What it is:** 2D Heatmap (X=Time, Y=Leverage Bucket) showing the distribution of risk.
**Interpretation:**
*   **Risk Buildup:** A "hot" red zone in the >20x bucket usually precedes a cascade.
*   **Heterogeneity:** Shows if risk is concentrated in high-leverage gamblers.

#### 6. Enhanced Liquidation Timeline 💀
**What it is:** Price chart overlay with markers for liquidation events.
*   **Yellow Dot:** Partial Liquidation (Soft).
*   **Red X:** Full Bankruptcy (Hard).
**Interpretation:** distinguishes between controlled deleveraging and catastrophic failure.

#### 7. Equity-at-Risk Snapshot 💰
**What it is:** Histogram of System Equity at the end of the simulation.
**Interpretation:**
*   **Left Skew:** Indicates high tail risk of insolvency.
*   **Red Line:** The $0 bankruptcy threshold.

#### 8. Risk Diamond (Trader Heterogeneity) 💎
**What it is:** Scatterplot of Position Size vs Leverage, sized by Margin Usage.
**Interpretation:**
*   **Top-Right:** Dangerous "Systemic Whales".
*   **Top-Left:** High-leverage "Gamblers".
*   **Bottom-Left:** Safe retail flow.

---

### 🛠️ Plotting Workflow

**1. Run Simulation**
```bash
dex-sim run config/aes_vs_fxd.yaml
```

**2. List Runs**
Find the ID of the run you just completed.
```bash
dex-sim list
```

**3. Regenerate Plots (Optional)**
If you change the plotting code or want to re-plot an old run without re-simulating:
```bash
dex-sim plot results/20231129_120000_aes_vs_fxd
```

**4. Customizing Plots**
To add new plots, edit `src/dex_sim/plotting.py`. The `plot_all` function drives the generation process. Any new function added there will automatically be included in the workflow.

## Development Setup

The project follows a standard Python project structure.

```
dex-sim/
├── config/                 # Experiment configuration files
├── data/                   # Input data
├── results/                # Simulation outputs
├── src/
│   └── dex_sim/
│       ├── __init__.py
│       ├── cli.py          # Command-line interface
│       ├── engine.py       # Core Numba simulation engine
│       ├── models/         # Risk model definitions
│       ├── plotting.py     # Plotting functions
│       └── ...
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

**Coding Standards**:
- Code should be formatted with `ruff format`.
- Linting is done with `ruff`.
- Follow standard Python naming conventions (PEP 8).