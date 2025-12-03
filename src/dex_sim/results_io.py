import os
import json
import zarr
import numpy as np

from .data_structures import SingleModelResults, MultiModelResults


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _save_array(store, key, arr):
    if arr is None:
        return
    store.array(key, arr, chunks=True, overwrite=True)


def _load_array(store, key):
    return store[key][...] if key in store else None


# ------------------------------------------------------------
# Save MultiModelResults -> Zarr directory
# ------------------------------------------------------------


def save_results(results: MultiModelResults, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Save global metadata
    meta = {
        "num_paths": results.num_paths,
        "horizon": results.horizon,
        "initial_price": results.initial_price,
        "notional": results.notional,
        "metadata": results.metadata,
    }
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Zarr root
    root = zarr.open_group(os.path.join(outdir, "data.zarr"), mode="w")

    # Optional MC inputs
    _save_array(root, "log_returns", results.log_returns)
    _save_array(root, "amihud_le", results.amihud_le)
    _save_array(root, "sigma_path", results.sigma_path)

    # Save each model in a sub-group
    models_grp = root.create_group("models")
    for name, model in results.models.items():
        g = models_grp.create_group(name)

        _save_array(g, "df_required", model.df_required)
        _save_array(g, "defaults", model.defaults)
        _save_array(g, "price_paths", model.price_paths)
        _save_array(g, "lev_long", model.lev_long)
        _save_array(g, "lev_short", model.lev_short)

        _save_array(g, "rt", model.rt)
        _save_array(g, "breaker_state", model.breaker_state)
        _save_array(g, "margin_multiplier", model.margin_multiplier)

        _save_array(g, "partial_liq_amount", model.partial_liq_amount)
        _save_array(g, "notional_paths", model.notional_paths)

        _save_array(g, "equity_long", model.equity_long)
        _save_array(g, "equity_short", model.equity_short)

        _save_array(g, "ecp_position_path", model.ecp_position_path)
        _save_array(g, "ecp_slippage_cost", model.ecp_slippage_cost)

        # Save metadata
        if model.metadata:
            with open(os.path.join(outdir, f"metadata_{name}.json"), "w") as f:
                json.dump(model.metadata, f, indent=2)


# ------------------------------------------------------------
# Load Zarr directory -> MultiModelResults
# ------------------------------------------------------------


def load_results(outdir: str) -> MultiModelResults:
    with open(os.path.join(outdir, "metadata.json"), "r") as f:
        meta = json.load(f)

    root = zarr.open_group(os.path.join(outdir, "data.zarr"), mode="r")

    # Load MC inputs
    log_returns = _load_array(root, "log_returns")
    amihud_le = _load_array(root, "amihud_le")
    sigma_path = _load_array(root, "sigma_path")

    # Load per-model groups
    models_grp = root["models"]
    models = {}

    for name in models_grp:
        g = models_grp[name]

        models[name] = SingleModelResults(
            name=name,
            df_required=_load_array(g, "df_required"),
            defaults=_load_array(g, "defaults"),
            price_paths=_load_array(g, "price_paths"),
            lev_long=_load_array(g, "lev_long"),
            lev_short=_load_array(g, "lev_short"),
            rt=_load_array(g, "rt"),
            breaker_state=_load_array(g, "breaker_state"),
            margin_multiplier=_load_array(g, "margin_multiplier"),
            partial_liq_amount=_load_array(g, "partial_liq_amount"),
            notional_paths=_load_array(g, "notional_paths"),
            equity_long=_load_array(g, "equity_long"),
            equity_short=_load_array(g, "equity_short"),
            ecp_position_path=_load_array(g, "ecp_position_path"),
            ecp_slippage_cost=_load_array(g, "ecp_slippage_cost"),
            metadata=(
                json.load(open(os.path.join(outdir, f"metadata_{name}.json")))
                if os.path.exists(os.path.join(outdir, f"metadata_{name}.json"))
                else {}
            ),
        )

    return MultiModelResults(
        models=models,
        num_paths=meta["num_paths"],
        horizon=meta["horizon"],
        initial_price=meta["initial_price"],
        notional=meta["notional"],
        log_returns=log_returns,
        amihud_le=amihud_le,
        sigma_path=sigma_path,
        metadata=meta["metadata"],
    )
