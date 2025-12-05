import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...data_structures import SingleModelResults
from ..layout import apply_standard_layout

def make_liquidation_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    # For liquidations, we might want dual axis?
    # But "Small multiples ... One subplot per model".
    # Plotly subplots with secondary_y per row is complex in loop.
    # Let's use bar for cost and line for fraction on same plot with different axis magnitude?
    # No, fraction is 0-1 (or small). Cost is $. 
    # We really need secondary y.
    
    rows = len(models)
    titles = list(models.keys())
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles, 
                        specs=[[{"secondary_y": True}]] * rows, vertical_spacing=0.05)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        if res.liquidation_fraction is None: continue
        
        # Mean across paths
        liq_frac = np.nanmean(res.liquidation_fraction, axis=0)
        slip_cost = np.nanmean(res.slippage_cost, axis=0)
        
        x = np.arange(len(liq_frac))
        
        # Cost on primary (Bar)
        fig.add_trace(
            go.Bar(x=x, y=slip_cost, name=f"{name} Cost ($)", marker_color='pink', opacity=0.6),
            row=row, col=1, secondary_y=False
        )
        
        # Fraction on secondary (Line)
        fig.add_trace(
            go.Scatter(x=x, y=liq_frac, name=f"{name} Fraction", line=dict(color='purple')),
            row=row, col=1, secondary_y=True
        )
        
    apply_standard_layout(fig, "Liquidation Intensity & Costs")
    return fig

def make_liquidation_panel(model: SingleModelResults) -> go.Figure:
    return go.Figure()
