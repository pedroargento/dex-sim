from .dashboard import make_dashboard
from .panels.risk_panel import make_risk_compare_panels, make_risk_panel
from .panels.leverage_panel import make_leverage_compare_panels, make_leverage_panel
from .panels.margin_panel import make_margin_compare_panels, make_margin_panel
from .panels.tradeflow_panel import make_tradeflow_compare_panels, make_tradeflow_panel
from .panels.liquidation_panel import make_liquidation_compare_panels, make_liquidation_panel
from .panels.exposure_panel import make_exposure_compare_panels, make_exposure_panel
from .panels.distribution_panel import make_distribution_compare_panels, make_distribution_panel
from .panels.path_drilldown_panel import make_path_drilldown_panel
from .utils.color_schemes import BREAKER_COLORS, TRACE_COLORS
from .dashboard_export import generate_dashboard

def plot_all(results, out_dir: str):
    """
    Legacy entry point for plotting.
    Redirects to the new Plotly dashboard generator.
    """
    generate_dashboard(results, out_dir)