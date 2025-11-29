#!/usr/bin/env python3

import numpy as np
import argparse
import os
from datetime import datetime

from dex_sim.engine import run_models_numba
from dex_sim.models import (
    AESModel,
    FXDModel,
    ES_IM,
    FixedLeverageIM,
    Breaker,
    FullCloseOut,
)
from dex_sim.results_io import save_results
from dex_sim.plotting import plot_all_for_model


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------


def run_simulation(args):
    np.random.seed(args.seed)

    # --- Define models (you can extend this arbitrarily) ---
    aes = AESModel(
        name="AES_es99",
        im=ES_IM(conf=0.99),
        breaker=Breaker(
            soft=1.0,
            hard=2.0,
            multipliers=(1.0, 1.5, 2.0),
        ),
        liquidation=FullCloseOut(slippage_factor=args.slippage),
    )

    fxd = FXDModel(
        name=f"FXD_{args.fxd_leverage}x",
        im=FixedLeverageIM(leverage=args.fxd_leverage),
        liquidation=FullCloseOut(slippage_factor=args.slippage),
    )

    models = [aes, fxd]

    # --- Run simulation ---
    print("\n=== Running Pluggable Risk Simulation ===")
    print(f"Paths: {args.paths}, Horizon: auto via GARCH (mc_generator)")
    print(f"Models: {[m.name for m in models]}")
    print("Using Numba ultra-fast engine...\n")

    results = run_models_numba(
        models=models,
        num_paths=args.paths,
        initial_price=args.initial_price,
        total_oi=args.total_oi,
        whale_oi_fraction=args.oi_fraction,
        stress_factor=args.stress_factor,
        garch_params_file=args.garch_params,
    )

    # --- Save results ---
    outdir = args.outdir or f"results/run_{timestamp()}"
    print(f"Saving results to: {outdir}")
    save_results(results, outdir)

    # --- Plot (optional) ---
    if args.plot:
        print("Generating charts...")
        for name, model_res in results.models.items():
            plot_all_for_model(
                model_res, outdir=os.path.join(outdir, "plots"), max_paths=5
            )

    print("\nDone.\n")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run pluggable risk simulation.")

    parser.add_argument(
        "--paths", type=int, default=5000, help="Number of Monte Carlo paths"
    )

    parser.add_argument("--initial-price", type=float, default=4000.0)
    parser.add_argument("--total-oi", type=float, default=1_000_000_000.0)
    parser.add_argument(
        "--oi-fraction",
        type=float,
        default=0.40,
        help="Whale OI fraction (defines notional)",
    )
    parser.add_argument("--fxd-leverage", type=float, default=20.0)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--stress-factor", type=float, default=1.0)

    parser.add_argument("--garch-params", type=str, default="garch_params.json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save results (default: timestamped)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plotting output")

    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
