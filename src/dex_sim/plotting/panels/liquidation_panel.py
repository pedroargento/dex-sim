import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...data_structures import SingleModelResults
from ..layout import apply_standard_layout

def make_liquidation_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    
    rows = len(models)
    titles = list(models.keys())
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles, 
                        specs=[[{"secondary_y": True}]] * rows, vertical_spacing=0.05)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        has_liq = res.liquidation_fraction is not None and res.liquidation_fraction.size > 0 and not np.all(np.isnan(res.liquidation_fraction))
        
        if not has_liq:
            fig.add_annotation(
                text=f"{name}: No liquidation data",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red"),
                secondary_y=False
            )
            continue
        
        # Mean across paths
        liq_frac = np.nanmean(res.liquidation_fraction, axis=0)
        
        has_cost = res.slippage_cost is not None and res.slippage_cost.size > 0 and not np.all(np.isnan(res.slippage_cost))
        if has_cost:
            slip_cost = np.nanmean(res.slippage_cost, axis=0)
        else:
            slip_cost = np.zeros_like(liq_frac)
        
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