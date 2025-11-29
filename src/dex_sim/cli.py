import argparse
from .experiment_manager import (
    run_experiment_from_config,
    list_experiments,
    plot_experiment,
    compare_experiments,
)


def main():
    parser = argparse.ArgumentParser(description="dex-sim experiment manager")
    sub = parser.add_subparsers(dest="cmd")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run an experiment from a YAML config")
    p_run.add_argument("config", type=str, help="Path to .yaml config file")

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    sub.add_parser("list", help="List previous experiment runs")

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------
    p_plot = sub.add_parser("plot", help="Plot all charts for a run")
    p_plot.add_argument("run_dir", type=str, help="Path to run output directory")

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="Compare DF distributions across runs")
    p_cmp.add_argument(
        "runs", nargs="+", help="Run IDs (directory names under results/)"
    )

    args = parser.parse_args()

    if args.cmd == "run":
        run_experiment_from_config(args.config)

    elif args.cmd == "list":
        list_experiments()

    elif args.cmd == "plot":
        plot_experiment(args.run_dir)

    elif args.cmd == "compare":
        compare_experiments(args.runs)

    else:
        parser.print_help()
