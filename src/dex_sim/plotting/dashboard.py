import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional

from ..data_structures import SingleModelResults, MultiModelResults
from .panels.risk_panel import make_risk_compare_panels
from .panels.leverage_panel import make_leverage_compare_panels
from .panels.margin_panel import make_margin_compare_panels
from .panels.tradeflow_panel import make_tradeflow_compare_panels
from .panels.liquidation_panel import make_liquidation_compare_panels
from .panels.exposure_panel import make_exposure_compare_panels
from .panels.distribution_panel import make_distribution_compare_panels


def make_dashboard(
    results: MultiModelResults,
    sigma_path: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Assemble an interactive Plotly dashboard using small multiples:
    one subplot (row) per model.

    Views (switched via dropdown):
      - Risk
      - Leverage
      - Margin
      - Tradeflow
      - Liquidation
      - Exposure
      - Distributions

    Implementation strategy:
      * Each panel module returns a Figure with n_rows = n_models.
      * All these figures use the same axis naming scheme (xaxis, xaxis2, ...).
      * We:
        - Pick one panel as the 'layout template' (Leverage).
        - Create a master Figure by copying that panel.
        - Append all other panels' traces to the master, but keep them invisible.
        - Build an updatemenu that toggles visibility of trace groups per view.
    """

    models: Dict[str, SingleModelResults] = results.models
    model_names = list(models.keys())
    n_models = len(model_names)

    # ------------------------------------------------------------------
    # 1. Build all comparison panels
    # ------------------------------------------------------------------
    panel_figs: Dict[str, go.Figure] = {}

    # Risk may have NaN-heavy data (e.g. models without rt). It is still
    # useful, but we won't make it the default visible view.
    panel_figs["Risk"] = make_risk_compare_panels(models)

    # Use Leverage as the layout template and default view
    panel_figs["Leverage"] = make_leverage_compare_panels(models)

    # Margin uses sigma_path when available
    panel_figs["Margin"] = make_margin_compare_panels(models, sigma_path)

    panel_figs["Tradeflow"] = make_tradeflow_compare_panels(models)
    panel_figs["Liquidation"] = make_liquidation_compare_panels(models)
    panel_figs["Exposure"] = make_exposure_compare_panels(models)
    panel_figs["Distributions"] = make_distribution_compare_panels(models)

    # ------------------------------------------------------------------
    # 2. Initialise master figure from the Leverage panel
    # ------------------------------------------------------------------
    base_fig = panel_figs["Leverage"]
    master_fig = go.Figure(base_fig)  # copies data + layout

    # Track trace index ranges per view
    views = []
    current_idx = 0

    # Leverage is first: its traces are already in master_fig
    n_lev = len(base_fig.data)
    views.append(("Leverage", 0, n_lev, base_fig.layout))
    current_idx += n_lev

    # ------------------------------------------------------------------
    # 3. Append remaining panel traces (all invisible by default)
    # ------------------------------------------------------------------
    def _add_panel(name: str):
        nonlocal current_idx
        fig = panel_figs[name]
        n = len(fig.data)
        if n == 0:
            return

        start = current_idx
        for tr in fig.data:
            # Ensure invisible until user selects this view
            tr.visible = False
            master_fig.add_trace(tr)
        end = current_idx + n
        current_idx = end

        views.append((name, start, end, fig.layout))

    # Order of views in dropdown
    for panel_name in [
        "Risk",
        "Margin",
        "Tradeflow",
        "Liquidation",
        "Exposure",
        "Distributions",
    ]:
        _add_panel(panel_name)

    total_traces = len(master_fig.data)

    # ------------------------------------------------------------------
    # 4. Build dropdown buttons: one per view
    # ------------------------------------------------------------------
    buttons = []

    for label, start, end, layout in views:
        visible_mask = [False] * total_traces
        for i in range(start, end):
            visible_mask[i] = True

        # Title fallback if missing
        title_text = (
            layout.title.text
            if hasattr(layout, "title") and layout.title.text
            else label
        )

        # We avoid trying to propagate full axis configs (too brittle across panels).
        # Only update the title when switching views.
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible_mask},
                    {"title": title_text},
                ],
            )
        )

    # By construction, Leverage traces (indices [0:n_lev]) are visible
    # and all others were added as visible=False, so the initial view is correct.

    master_fig.update_layout(
        updatemenus=[
            dict(
                active=0,  # Leverage
                buttons=buttons,
                direction="down",
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        title=f"Sim Dashboard ({n_models} models)",
    )

    return master_fig
