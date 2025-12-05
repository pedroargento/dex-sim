import os
from ..data_structures import MultiModelResults
from .dashboard import make_dashboard
from .panels.path_drilldown_panel import make_path_drilldown_panel


def generate_dashboard(sim: MultiModelResults, out_dir: str) -> str:
    """
    Generates an interactive Plotly dashboard for all models.
    Saves:
        - dashboard.html  (comparison views)
        - drilldown.html  (path-level explorer)
    Returns:
        Path to dashboard.html
    """
    # sigma_path is provided by MultiModelResults
    sigma_path = getattr(sim, "sigma_path", None)

    # NEW: pass entire sim object (not sim.models)
    fig = make_dashboard(sim, sigma_path=sigma_path)

    dashboard_path = os.path.join(out_dir, "dashboard.html")
    fig.write_html(dashboard_path, include_plotlyjs="cdn")

    # Drilldown still works on per-model dict
    drill = make_path_drilldown_panel(sim.models)
    drilldown_path = os.path.join(out_dir, "drilldown.html")
    drill.write_html(drilldown_path, include_plotlyjs="cdn")

    return dashboard_path
