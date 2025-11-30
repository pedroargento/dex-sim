import os
import json
import yaml
from datetime import datetime

import numpy as np

from .engine import run_models_numba
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
        soft = float(cfg.get("soft", float("inf"))),
        hard = float(cfg.get("hard", float("inf"))),
        multipliers = tuple(cfg.get("multipliers", [1.0, 1.0, 1.0])),
    )

def build_liquidation(cfg):
    t = cfg.get("type", "full")
    if t == "full":
        return FullCloseOut(slippage_factor=cfg.get("slippage", 0.001))
    elif t == "partial":
        return PartialCloseOut(slippage_factor=cfg.get("slippage", 0.001))
    else:
        raise ValueError(f"Unknown liquidation type: {t}")

# ------------------------------------------------------------
# Model factory (reads YAML model definitions)
# ------------------------------------------------------------


def build_model(mcfg: dict):
    """Build a RiskModel from a YAML model config."""
    return RiskModel(
        name = mcfg["name"],
        im = build_im(mcfg["im"]),
        breaker = build_breaker(mcfg.get("breaker", {})),
        liquidation = build_liquidation(mcfg.get("liquidation", {})),
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

    # Notional: NEW API
    notional = cfg.get("notional")
    if notional is None:
        # Backward compatibility: allow total_oi * oi_fraction, but warn
        total_oi = cfg.get("total_oi")
        oi_fraction = cfg.get("oi_fraction")
        if total_oi is not None and oi_fraction is not None:
            print(
                "[WARN] `total_oi` + `oi_fraction` are deprecated. "
                "Please specify `notional` directly in the config."
            )
            notional = float(total_oi) * float(oi_fraction)
        else:
            raise ValueError(
                "Config must include `notional` (position size). "
                "Old `total_oi`+`oi_fraction` interface is deprecated."
            )

    # partial_liquidation is no longer a global flag
    # It is determined per-model by its liquidation component

    np.random.seed(seed)

    print("\n=== Running Experiment ===")
    print(f"Config: {config_file}")
    print(f"Run ID: {rid}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Paths: {num_paths}")
    print(f"Notional per side: {notional:,.0f}")
    print(f"Stress factor: {stress_factor}")
    print()

    # Run simulation
    results = run_models_numba(
        models=models,
        num_paths=num_paths,
        initial_price=initial_price,
        notional=notional,
        stress_factor=stress_factor,
        garch_params_file=garch_params,
    )

    # Save results + metadata
    print(f"Saving results → {outdir}")
    save_results(results, outdir)

    # Generate Summary Report
    report_path = generate_summary(results, outdir)
    print(f"Summary report → {report_path}")

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