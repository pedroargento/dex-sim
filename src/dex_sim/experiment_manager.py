import os
import json
import yaml
from datetime import datetime

import numpy as np

from .engine import run_models
from .models import (
    RiskModel,
    ES_IM,
    FixedLeverageIM,
    Breaker,
    FullCloseOut,
    PartialCloseOut,
)
from .results_io import save_results, load_results
from .plotting import plot_all
from .summary import generate_summary


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def discover_runs(root: str = "results"):
    if not os.path.exists(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


# ------------------------------------------------------------
# Component Builders
# ------------------------------------------------------------


def build_im(cfg):
    t = cfg.get("type")
    if t == "es":
        return ES_IM(conf=cfg.get("conf", 0.99))
    elif t == "fixed_leverage":
        return FixedLeverageIM(leverage=cfg.get("leverage", 1.0))
    else:
        raise ValueError(f"Unknown IM type: {t}")


def build_breaker(cfg):
    # Default to infinite limits (no breaker) if not specified
    return Breaker(
        soft=float(cfg.get("soft", float("inf"))),
        hard=float(cfg.get("hard", float("inf"))),
        multipliers=tuple(cfg.get("multipliers", [1.0, 1.0, 1.0])),
    )


def build_liquidation(cfg):

    t = cfg.get("type", "full")

    if t == "full":

        return FullCloseOut(slippage_factor=cfg.get("slippage", 0.001))

    elif t == "partial":

        return PartialCloseOut(slippage_factor=cfg.get("slippage", 0.001))

    else:

        raise ValueError(f"Unknown liquidation type: {t}")
def build_trader_pool(cfg: dict):
    """
    Construct fixed trader pool arrays.
    Symmetric population: N must be even.
    0..N/2-1: Primary
    N/2..N-1: Mirror
    """
    traders_cfg = cfg.get("traders", {})
    
    # Backward compatibility / Default
    N = int(traders_cfg.get("count", cfg.get("num_traders", 2000)))
    
    if N % 2 != 0:
        print(f"[WARN] Trader count {N} is odd. Increasing to {N+1} for symmetry.")
        N += 1

    # Warn if old 'trader_arrival' is present
    if "trader_arrival" in cfg:
        print("[WARN] 'trader_arrival' config is deprecated and ignored. Using fixed trader pool.")

    if "max_leverage" in traders_cfg:
         print("[WARN] 'max_leverage' in trader config is deprecated and ignored. Leverage is governed by IM/MM.")

    # Initialize arrays
    # pool_arrival_tick is irrelevant now, all start at 0
    pool_arrival_tick = np.zeros(N, dtype=np.int64)
    
    # Initial Notional is 0. All traders start flat.
    # We keep the array for kernel compatibility but zero it out.
    pool_notional = np.zeros(N, dtype=np.float64)
    
    # Direction is implicit in symmetry logic (Primary vs Mirror), 
    # but we can keep an array if needed. For now, zeros.
    pool_direction = np.zeros(N, dtype=np.float64)
    
    # Equity
    init_equity = float(traders_cfg.get("initial_equity", 10000.0))
    pool_equity = np.full(N, init_equity, dtype=np.float64)
    
    # Behaviors
    # Config: behaviors: { expand_fraction: 0.5 }
    behaviors_cfg = traders_cfg.get("behaviors", {})
    expand_frac = float(behaviors_cfg.get("expand_fraction", 0.5))
    
    pool_behavior_id = np.zeros(N, dtype=np.int64) 
    
    # Assign behaviors to Primary traders (0 to N/2 - 1)
    half_N = N // 2
    num_expanders_half = int(half_N * expand_frac)
    
    # 0 = Expander, 1 = Reducer
    # First num_expanders_half get 0, rest get 1
    # Mirror traders (N/2 to N-1) copy their primary counterpart
    
    for i in range(half_N):
        bid = 0 if i < num_expanders_half else 1
        pool_behavior_id[i] = bid
        pool_behavior_id[i + half_N] = bid
    
    return (
        pool_arrival_tick,
        pool_notional,
        pool_direction,
        pool_equity,
        pool_behavior_id
    )


# ------------------------------------------------------------

# Model factory (reads YAML model definitions)

# ------------------------------------------------------------


def build_model(mcfg: dict):
    """Build a RiskModel from a YAML model config."""
    return RiskModel(
        name=mcfg["name"],
        im=build_im(mcfg["im"]),
        breaker=build_breaker(mcfg.get("breaker", {})),
        liquidation=build_liquidation(mcfg.get("liquidation", {})),
        backend=mcfg.get("backend", "python"),
        gamma=float(mcfg.get("gamma", 0.8)),
    )


# ------------------------------------------------------------
# Run experiment defined by YAML config
# ------------------------------------------------------------


def run_experiment_from_config(config_file: str, root: str = "results") -> str:
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("name", "experiment")
    rid = f"{now_id()}_{exp_name}"
    outdir = os.path.join(root, rid)
    ensure_dir(outdir)

    # Build models
    models = [build_model(m) for m in cfg["models"]]

    # Simulation parameters
    num_paths = cfg.get("paths", 5000)
    initial_price = cfg.get("initial_price", 4000.0)
    stress_factor = cfg.get("stress_factor", 1.0)
    seed = cfg.get("seed", 42)
    garch_params = cfg.get("garch_params", "garch_params.json")

    if "notional" in cfg:
        print("[WARN] 'notional' in global config is deprecated. OI emerges endogenously.")

    # Build Trader Pool
    pool_data = build_trader_pool(cfg)

    np.random.seed(seed)

    print("\n=== Running Experiment ===")
    print(f"Config: {config_file}")
    print(f"Run ID: {rid}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Paths: {num_paths}")
    print(f"Stress factor: {stress_factor}")
    print(f"Traders: {len(pool_data[0])} (Symmetric Pool)")
    print()

    # Run simulation
    results = run_models(
        models=models,
        trader_pool=pool_data,
        trader_pool_config=cfg.get("traders", {}),
        num_paths=num_paths,
        initial_price=initial_price,
        stress_factor=stress_factor,
        garch_params_file=garch_params,
    )

    # Save results + metadata
    print(f"Saving results → {outdir}")
    save_results(results, outdir)

    # Generate Summary Report
    summary_path = generate_summary(results, outdir)
    print(f"Summary report → {summary_path}")

    # Generate Plotly Dashboard
    from dex_sim.plotting.dashboard_export import generate_dashboard
    dashboard_path = generate_dashboard(results, outdir)
    print(f"Interactive dashboard → {dashboard_path}")

    with open(os.path.join(outdir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    print("Done.")
    return outdir


# ------------------------------------------------------------
# Plotting an experiment
# ------------------------------------------------------------


def plot_experiment(run_dir: str):
    print(f"Loading results from: {run_dir}")
    results = load_results(run_dir)

    outdir = os.path.join(run_dir, "plots")
    ensure_dir(outdir)

    print("Generating visualization suite...")
    plot_all(results, outdir)

    print("Plots saved in:", outdir)


# ------------------------------------------------------------
# Compare experiments (simple DF comparison)
# ------------------------------------------------------------


def compare_experiments(runs: list[str], root: str = "results"):
    loaded = {}
    for rd in runs:
        path = os.path.join(root, rd)
        print(f"Loading {path}...")
        loaded[rd] = load_results(path)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compare DF distributions for matching model names
    base_run = runs[0]
    for model_name in loaded[base_run].models.keys():
        plt.figure(figsize=(8, 5))
        for rd in runs:
            df = loaded[rd].models[model_name].df_required
            sns.histplot(df, label=rd, kde=False, stat="density", alpha=0.6, bins=50)

        plt.legend()
        plt.title(f"DF Distribution Comparison — Model: {model_name}")
        plt.xlabel("DF Required ($)")
        plt.tight_layout()
        outpath = os.path.join(root, f"compare_{model_name}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"Saved comparison plot for {model_name} → {outpath}")

    print("Comparison plots saved.")


# ------------------------------------------------------------
# List all runs
# ------------------------------------------------------------


def list_experiments(root: str = "results"):
    runs = discover_runs(root)
    print("\n=== Available Experiment Runs ===")
    if not runs:
        print("(none)")
        return
    for r in runs:
        print(" •", r)