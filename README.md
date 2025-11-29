#  dex-sim: DEX - CCP Risk Simulation

`dex-sim` is a high-performance risk simulation framework for modeling Central Counterparty (CCP) style margining, liquidations, and systemic stress for perpetual futures exchanges.

It is designed for speed, accuracy, and flexibility, allowing researchers and developers to test and compare different risk management models under extreme market conditions.

## Project Overview

`dex-sim` simulates the financial risk dynamics of a perpetual futures exchange operating under a CCP model. The core of the project is a powerful Monte Carlo engine that generates thousands of market scenarios to assess how different risk models perform.

- **Risk Modeling**: Supports multiple pluggable risk models, including the innovative **Adaptive Exposure System (AES)**, traditional **Fixed Leverage (FXD)**, and easily extensible custom models.
- **High-Performance Engine**: Utilizes a **Numba-accelerated** Monte Carlo engine to simulate thousands of price paths with GARCH-based volatility models, ensuring both speed and realism.
- **Systemic Stress Testing**: Models complex dynamics like circuit breakers, dynamic margin adjustments, and default fund consumption.

## Features

- **Pluggable Risk Models**: Easily define and switch between different margin and liquidation systems.
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

## Installation

Get up and running in a few simple steps. `dex-sim` uses `uv` for fast and reliable dependency management.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/dex-sim.git

# 2. Navigate to the project directory
cd dex-sim

# 3. Install in editable mode
uv pip install -e .
```

## Quick Start

Running simulations and analyzing results is simple with the built-in CLI.

**1. Run an Experiment**

Experiments are defined in YAML files. A sample configuration is provided in `config/aes_vs_fxd_test.yaml`.

```bash
dex-sim run config/aes_vs_fxd_test.yaml
```
This command will run the simulation, save the results, and generate plots.

**2. List Simulation Runs**

You can list all the completed simulation runs stored in the `results/` directory.

```bash
dex-sim list
```

**3. Plot Results**

Generate plots for a specific run using its unique ID.

```bash
dex-sim plot results/<run_id>
```

## Experiment Configuration

Experiments are defined in a simple YAML format, allowing you to configure the simulation parameters and the risk models to be tested.

Here is a working example from `config/aes_vs_fxd_test.yaml`:

```yaml
name: aes_vs_fxd_test
paths: 5000
initial_price: 4000
notional: 400000000
stress_factor: 1.0
garch_params: garch_params.json

models:
  - type: AES
    name: AES_es99
    im_conf: 0.99
    slippage: 0.001
    breaker_soft: 1.0
    breaker_hard: 2.0
    breaker_mult: [1.0, 1.5, 2.0]

  - type: FXD
    name: FXD_20x
    leverage: 20
    slippage: 0.001
```

## Risk Models

`dex-sim` supports different risk models, each with its own approach to margining and liquidation.

### AES (Adaptive Exposure System)

The **AES** model implements a dynamic margining system. Initial Margin is based on a high-confidence **Expected Shortfall (ES)** of future returns. During the simulation, the system responds to market stress (measured by **Rₜ**) by activating circuit breaker regimes (SOFT, HARD) that increase margin multipliers, automatically tightening leverage when risk is high.

### FXD (Fixed Leverage)

The **FXD** model represents a traditional exchange where traders are offered a fixed leverage level (e.g., 20x). This model serves as a baseline to demonstrate the performance of adaptive models like AES under stress.

### Slippage and Closeout

Both models use a **Full Closeout** liquidation mechanism. When a default occurs, the model assumes the entire position is closed at a cost, which is modeled as a `slippage_factor` applied to the position's notional value. This cost is drawn from the default fund.

## Simulation Engine

The heart of `dex-sim` is its Numba JIT-compiled simulation core.

- **Numba JIT Core**: The core loop is written in a way that allows Numba to translate it into highly efficient machine code, enabling the simulation of millions of timesteps in seconds.
- **2-Trader Symmetric Accounting**: The simulation simplifies the market into a symmetric two-trader model (one long, one short) with equal notional exposure. Variation Margin (VM) is calculated as the PnL transfer between them.
- **Default Fund Logic**: If a trader's equity is insufficient to cover their losses (VM), they default. The remaining loss, plus slippage, is covered by the Default Fund (DF).
- **Dynamic Margins**: For adaptive models like AES, the systemic risk index **Rₜ** is calculated at each timestep. If Rₜ crosses the breaker thresholds, the margin multipliers are increased for that path, affecting the amount of capital drawn from the DF in a default.

##  Using the Results

All simulation outputs are saved to `results/<run_id>/data.zarr`, where `<run_id>` is a timestamped identifier for your experiment. The Zarr format allows for efficient storage and retrieval of large numerical arrays.

The output data includes:
- `price_paths`: The simulated price for each path and timestep.
- `df_required`: The total amount required from the Default Fund for each path.
- `leverage`: The effective leverage for the long and short trader at each timestep.
- `breaker_state`: An integer (0, 1, or 2) indicating the circuit breaker state (NORMAL, SOFT, HARD) at each timestep.
- `rt`: The value of the systemic risk index Rₜ at each timestep.

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
*   **AES vs FXD:** Compare the tails. Does AES drop off faster (safer) or does it have the same extreme tail risk as FXD?

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

## Creating and Running Custom Models

`dex-sim` is built to be extensible. You can easily create, configure, and run your own risk models. Here’s a step-by-step guide.

### 1. Define Your Custom Model

First, create a new Python file in the `src/dex_sim/models/` directory (e.g., `my_model.py`). Inside this file, define your model as a class that inherits from `RiskModel`.

Your class must:
1.  Inherit from `dex_sim.models.base.RiskModel`.
2.  Implement the `initial_margin(self, notional: float, sigma: float) -> float` method via a component.
3.  Accept a `name` argument in its `__init__` method and pass it to the parent class.

You can also add other components like `breaker` or `liquidation` logic, and accept custom parameters in `__init__`.

**Example: `src/dex_sim/models/my_model.py`**

Let's create a `SimpleVaRModel` that calculates Initial Margin based on a Value-at-Risk (VaR) formula with a configurable confidence level and scaling factor.

```python
# src/dex_sim/models/my_model.py

from .base import RiskModel, IM_Component, Liquidation_Component

class SimpleVaR_IM(IM_Component):
    """A simple VaR-based Initial Margin component."""
    def __init__(self, var_conf: float = 0.99, scaler: float = 1.2):
        self.var_conf = var_conf
        self.scaler = scaler

    def initial_margin(self, notional: float, sigma: float) -> float:
        # A simplified VaR-like calculation
        # Z-score for 99% confidence is approx 2.33
        z_score = 2.33 
        margin = z_score * sigma * notional
        return self.scaler * margin

class SimpleVaRModel(RiskModel):
    """
    A custom risk model using our SimpleVaR IM component.
    """
    def __init__(self, name: str, im: IM_Component, liquidation: Liquidation_Component):
        super().__init__(name=name, im=im, liquidation=liquidation)

```

### 2. Register Your Model

To make your model accessible to the experiment runner, you need to register it.

Open `src/dex_sim/experiment_manager.py` and:
1.  Import your new model class (`SimpleVaRModel`) and its components (`SimpleVaR_IM`).
2.  Add your model's `type` name to the `MODELS` dictionary, mapping it to your class.
3.  Add your new `IM_Component` to the `IM_COMPONENTS` dictionary.

**Example: `src/dex_sim/experiment_manager.py`**

```python
# ... other imports
from .models.aes import AESModel
from .models.fxd import FXDModel
# Import your new model and its components
from .models.my_model import SimpleVaRModel, SimpleVaR_IM 

# ...

# Add your model to the MODELS dictionary
MODELS = {
    "AES": AESModel,
    "FXD": FXDModel,
    "SimpleVaR": SimpleVaRModel, # <-- Register your new model class
}

# Add your IM component to the IM_COMPONENTS dictionary
IM_COMPONENTS = {
    "ES": ES_IM,
    "FixedLeverage": FixedLeverageIM,
    "SimpleVaR": SimpleVaR_IM, # <-- Register your new IM component class
}

# ...
```
*Note: This registration step is required to map the string `type` from the YAML file to the actual Python class.*

### 3. Configure Your Experiment

Now you can use your `SimpleVaR` model in a YAML experiment file. Create a new configuration or modify an existing one (e.g., `config/my_experiment.yaml`).

In the `models` section, add a new entry with `type: SimpleVaR`. You can configure the parameters you defined in `SimpleVaR_IM` (`var_conf` and `scaler`) and `FullCloseOut` (`slippage`) directly. The experiment manager will automatically construct the objects.

**Example: `config/my_experiment.yaml`**

```yaml
name: custom_var_vs_aes
paths: 10000
initial_price: 4000
notional: 400000000
stress_factor: 1.2
garch_params: garch_params.json

models:
  # Your custom model configuration
  - type: SimpleVaR
    name: SimpleVaR_99
    slippage: 0.001
    im:
      type: SimpleVaR # Specify the IM component type
      var_conf: 0.99
      scaler: 1.25

  # An AES model for comparison
  - type: AES
    name: AES_es99
    im_conf: 0.99
    slippage: 0.001
    breaker_soft: 1.0
    breaker_hard: 2.0
    breaker_mult: [1.0, 1.5, 2.0]
```

### 4. Run and Analyze

Finally, run your new experiment using the CLI and analyze the results.

```bash
# Run the simulation with your custom model
dex-sim run config/my_experiment.yaml

# List the run
dex-sim list

# Plot the results for your specific experiment
dex-sim plot results/run_<run_id_of_your_experiment>
```

Your custom model's performance will be simulated, saved, and plotted alongside any other models you included in the experiment, allowing for direct comparison.

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