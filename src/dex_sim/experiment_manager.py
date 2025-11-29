import os
import json
import yaml
import time
from datetime import datetime

import numpy as np

from .engine import run_models_numba
from .models import (
    AESModel,
    FXDModel,
    ES_IM,
    FixedLeverageIM,
    Breaker,
    FullCloseOut,
)
from .results_io import save_results, load_results
from .plotting import plot_all_for_model


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def discover_runs(root="results"):
    """Return list of available result directories."""
    if not os.path.exists(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


# ------------------------------------------------------------
# Model factory (reads YAML model definitions)
# ------------------------------------------------------------


def build_model(mcfg: dict):
    """Build a RiskModel from YAML config."""
    mtype = mcfg["type"]

    if mtype == "AES":
        return AESModel(
            name=mcfg.get("name", "AES"),
            im=ES_IM(conf=mcfg.get("im_conf", 0.99)),
            breaker=Breaker(
                soft=mcfg.get("breaker_soft", 1.0),
                hard=mcfg.get("breaker_hard", 2.0),
                multipliers=tuple(mcfg.get("breaker_mult", [1.0, 1.5, 2.0])),
            ),
            liquidation=FullCloseOut(slippage_factor=mcfg.get("slippage", 0.001)),
        )

    elif mtype == "FXD":
        return FXDModel(
            name=mcfg.get("name", f"FXD_{mcfg['leverage']}x"),
            im=FixedLeverageIM(leverage=mcfg["leverage"]),
            liquidation=FullCloseOut(slippage_factor=mcfg.get("slippage", 0.001)),
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")


# ------------------------------------------------------------
# Run experiment defined by YAML config
# ------------------------------------------------------------


def run_experiment_from_config(config_file: str, root="results"):
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("name", "experiment")
    rid = f"{now_id()}_{exp_name}"
    outdir = os.path.join(root, rid)
    ensure_dir(outdir)

    # -------------------------
    # Build models
    # -------------------------
    models = []
    for m in cfg["models"]:
        models.append(build_model(m))

    # -------------------------
    # Extract simulation params
    # -------------------------
    num_paths = cfg.get("paths", 5000)
    initial_price = cfg.get("initial_price", 4000.0)
    total_oi = cfg.get("total_oi", 1_000_000_000.0)
    oi_fraction = cfg.get("oi_fraction", 0.40)
    stress_factor = cfg.get("stress_factor", 1.0)
    seed = cfg.get("seed", 42)
    garch_params = cfg.get("garch_params", "garch_params.json")

    np.random.seed(seed)

    print("\n=== Running Experiment ===")
    print(f"Config: {config_file}")
    print(f"Run ID: {rid}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Paths: {num_paths}")
    print(f"Stress factor: {stress_factor}")
    print()

    # -------------------------
    # Run simulation
    # -------------------------
    results = run_models_numba(
        models=models,
        num_paths=num_paths,
        initial_price=initial_price,
        total_oi=total_oi,
        whale_oi_fraction=oi_fraction,
        stress_factor=stress_factor,
        garch_params_file=garch_params,
    )

    # -------------------------
    # Save results + metadata
    # -------------------------
    print(f"Saving results → {outdir}")
    save_results(results, outdir)

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

    print("Plotting models...")
    for name, res in results.models.items():
        print(f" → {name}")
        plot_all_for_model(res, outdir=outdir, max_paths=5)

    print("Plots saved in:", outdir)


# ------------------------------------------------------------
# Compare experiments (simple DF comparison)
# ------------------------------------------------------------


def compare_experiments(runs: list, root="results"):
    loaded = {}
    for rd in runs:
        path = os.path.join(root, rd)
        print(f"Loading {path}...")
        loaded[rd] = load_results(path)

    # Compare DF distributions for matching model names
    import matplotlib.pyplot as plt

    for model_name in loaded[runs[0]].models.keys():
        plt.figure(figsize=(8, 5))
        for rd in runs:
            df = loaded[rd].models[model_name].df_required
            sns = __import__("seaborn")
            sns.histplot(df, label=rd, kde=False, stat="density")

        plt.legend()
        plt.title(f"DF Distribution Comparison — Model: {model_name}")
        plt.xlabel("DF Required ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(root, f"compare_{model_name}.png"))
        plt.close()

    print("Comparison plots saved.")


# ------------------------------------------------------------
# List all runs
# ------------------------------------------------------------


def list_experiments(root="results"):
    runs = discover_runs(root)
    print("\n=== Available Experiment Runs ===")
    if not runs:
        print("(none)")
        return
    for r in runs:
        print(" •", r)
