#!/usr/bin/env python3

import argparse
import os
import numpy as np
from datetime import datetime

from .engine_numba import run_models_numba
from .models import (
    AESModel,
    FXDModel,
    ES_IM,
    FixedLeverageIM,
    Breaker,
    FullCloseOut,
)
from .results_io import save_results
from .plotting import plot_all_for_model


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_simulation(args):
    np.random.seed(args.seed)

    # --- Define models ---
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

    print("\n=== Running Pluggable Risk Simulation ===")
    print(f"Paths: {args.paths}")
    print(f"Notional per side: {args.notional:,.0f}")
    print(f"Initial price: {args.initial_price}")
    print(f"Models: {[m.name for m in models]}")
    print("Engine: Numba\n")

    results = run_models_numba(
        models=models,
        num_paths=args.paths,
        initial_price=args.initial_price,
        notional=args.notional,
        stress_factor=args.stress_factor,
        garch_params_file=args.garch_params,
    )

    outdir = args.outdir or f"results/run_{timestamp()}"
    print(f"Saving results to: {outdir}")
    save_results(results, outdir)

    if args.plot:
        print("Generating charts...")
        for name, model_res in results.models.items():
            plot_all_for_model(
                model_res,
                outdir=os.path.join(outdir, "plots"),
                max_paths=5,
            )

    print("\nDone.\n")


def main():
    parser = argparse.ArgumentParser(description="Run pluggable risk simulation.")

    parser.add_argument(
        "--paths", type=int, default=5000, help="Number of Monte Carlo paths"
    )
    parser.add_argument(
        "--initial-price", type=float, default=4000.0, help="Starting spot price"
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=400_000_000.0,
        help="Position notional per side (long and short)",
    )
    parser.add_argument(
        "--fxd-leverage",
        type=float,
        default=20.0,
        help="Target fixed leverage for FXD model",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.001, help="Slippage factor for close-out"
    )
    parser.add_argument(
        "--stress-factor", type=float, default=1.0, help="GARCH stress scaling factor"
    )
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
