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

`dex-sim` uses a fully modular architecture. There is only one `RiskModel` class. A model is simply a composition of three pluggable components defined in YAML:

1.  **Initial Margin (IM)**: Determines the base margin requirement (e.g., `ES_IM` for dynamic risk, `FixedLeverageIM` for constant leverage).
2.  **Breaker**: A Finite State Machine that monitors systemic stress ($R_t$) and applies multipliers to the IM requirement.
3.  **Liquidation**: Defines how positions are closed when margin is breached (e.g., `FullCloseOut`, `PartialCloseOut`).

This architecture allows you to simulate almost any exchange design without writing new Python code.

## YAML Model Specification

Models are defined in the `models` list of your experiment configuration file.

```yaml
models:
  - name: MyCustomModel
    im:
      type: es              # 'es' or 'fixed_leverage'
      conf: 0.99            # Component-specific params
    breaker:
      soft: 1.0             # Trigger threshold for Soft state
      hard: 2.0             # Trigger threshold for Hard state
      multipliers: [1.0, 1.5, 2.0]  # [Normal, Soft, Hard] multipliers
    liquidation:
      type: partial         # 'full' or 'partial'
      slippage: 0.001
```

**Defaults:**
*   If `breaker` is omitted: Infinite thresholds (never triggers), multipliers `[1, 1, 1]`.
*   If `liquidation` is omitted: `type: full`, `slippage: 0.001`.

### Example 1: CCP-Like Model (Cartesi CCP)
Uses dynamic Expected Shortfall margin, an active circuit breaker, and partial liquidation to mitigate cascades.

```yaml
models:
  - name: CCP_ES99
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
    im: { type: es, conf: 0.99 }
    breaker: { soft: 0.4, hard: 0.7, multipliers: [1.0, 1.1, 1.25] }
    liquidation: { type: partial }

  - name: HL_20x
    im: { type: fixed_leverage, leverage: 20 }
    breaker: { soft: .inf, hard: .inf, multipliers: [1,1,1] }
    liquidation: { type: full }

  - name: Synthetic_10x
    im: { type: fixed_leverage, leverage: 10 }
    breaker: { soft: 0.5, hard: 0.9, multipliers: [1.0,1.2,1.4] }
    liquidation: { type: partial }
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

#### 2. Model Comparison Violins 🎻
**What it is:** Side-by-side violin plots comparing the density of losses between models.
**Interpretation:**
*   **Width:** The width of the shape represents the frequency of losses at that size.
*   **Base vs. Neck:** A wide base means frequent small losses (slippage). A long, thin neck means rare, massive bankruptcies.
*   **Risk Signal:** Ideally, you want a short, squat shape. Long necks indicate dangerous "Black Swan" potential.

#### 3. Efficiency Frontier (Scatter) ⚖️
**What it is:** A scatter plot of **Safety** (Max DF Usage) vs. **Capital Efficiency** (Average Margin Multiplier).
**Interpretation:**
*   **Goal:** The "holy grail" is the bottom-left corner: low margin requirements (high efficiency) AND low DF usage (high safety).
*   **Trade-off:** Usually, you trade one for the other. This chart quantifies exactly how much safety you buy with higher margins.

#### 4. Monte Carlo Convergence 🎯
**What it is:** A line chart showing the running average of DF usage as the number of simulation paths increases.
**Interpretation:**
*   **Stability:** If the line is still oscillating wildly at the end, your simulation needs more paths (increase `paths` in config). If it flattens out, your results are statistically significant.

---

### 🔬 Model Deep-Dive Charts (Per-Model)

For each model (e.g., `AES`, `FXD`), a dedicated folder is created with detailed forensics.

#### 5. Regime Dynamics Autopsy (Composite) 🔍
**What it is:** A 3-panel vertically stacked chart for a single "stress" path.
1.  **Top:** Asset Price & Systemic Risk Index ($R_t$).
2.  **Middle:** Breaker State (Shaded bands: Green=Normal, Orange=Soft, Red=Hard).
3.  **Bottom:** Margin Multiplier.
**Interpretation:**
*   **Causality:** Trace the chain reaction: Price Crash ➔ $R_t$ Spike ➔ Breaker Trigger ➔ Margin Hike.
*   **Lag:** Check if the breaker triggers *before* the worst losses or *after* (too late).

#### 6. Liquidation Intensity Heatmap 🔥
**What it is:** A dense heatmap where X=Time, Y=Path (sorted by severity), and Color=Liquidation Fraction ($k$).
**Interpretation:**
*   **Vertical Stripes:** A "systemic event" where the entire market liquidates simultaneously (correlation = 1).
*   **Gradient:** Shows the difference between a gentle partial liquidation (light red) and a hard closeout (dark red).
*   **Clustering:** Helps identify if liquidations are idiosyncratic (random dots) or structural (bands).

#### 7. Notional Decay Fan Chart 📉
**What it is:** A percentiles chart showing how the total Open Interest (notional) of the system decays over time.
**Interpretation:**
*   **De-leveraging:** Shows how fast the system reduces risk.
*   **Liquidity Crisis:** A vertical drop means the system is dumping massive inventory into the market instantly (high slippage risk). A gradual slope indicates a controlled "soft landing."

#### 8. Slippage Waterfall (Cost Composition) 💸
**What it is:** A breakdown of where the Default Fund money went.
**Interpretation:**
*   **Slippage:** Costs incurred by closing positions in illiquid markets.
*   **Bankruptcy:** Costs incurred because the trader ran out of money before liquidation could happen (gap risk).
*   **Signal:** High slippage means you need better liquidation logic. High bankruptcy means you need higher Initial Margin (IM).

#### 9. Worst-Case Autopsy 🚑
**What it is:** A detailed reconstruction of the single worst loss event in the entire simulation.
**Interpretation:**
*   **Narrative:** Tells the story of the failure. Did the trader die from a 1000-cut slow bleed, or one massive gap move?
*   **Equity:** Shows exactly when the trader's equity crossed zero (insolvency).

---

### 🧠 How to Read These Charts (Risk Analyst Guide)

*   **Look for Correlations:** Does a spike in $R_t$ reliably predict a cluster of liquidations? If not, your breaker sensitivity might be too low.
*   **Assess Tail Thickness:** In the **Survival Curve**, if the AES line is significantly below the FXD line in the bottom-right quadrant, AES is effectively mitigating catastrophic risk.
*   **Check De-leveraging Speed:** In the **Notional Fan Chart**, you want to see the system reducing exposure *before* the price crashes to zero. If notional stays flat while price drops, the system is too slow to react.
*   **Evaluate Efficiency:** If AES achieves the same safety (DF usage) as FXD but with a lower average margin multiplier (in the **Efficiency Frontier**), it proves AES is a superior capital-efficient model.

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