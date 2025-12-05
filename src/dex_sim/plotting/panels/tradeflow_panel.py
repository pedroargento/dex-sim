import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..layout import make_stacked_subplots, apply_standard_layout

def make_tradeflow_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        # Compute mean across paths
        if res.intent_accepted_normal is None: continue
        
        norm = np.nanmean(res.intent_accepted_normal, axis=0)
        red = np.nanmean(res.intent_accepted_reduce, axis=0)
        rej = np.nanmean(res.intent_rejected, axis=0)
        
        x = np.arange(len(norm))
        
        fig.add_trace(go.Bar(x=x, y=norm, name="Normal Accepted", marker_color='green'), row=row, col=1)
        fig.add_trace(go.Bar(x=x, y=red, name="Reduce Accepted", marker_color='orange'), row=row, col=1)
        fig.add_trace(go.Bar(x=x, y=rej, name="Rejected", marker_color='red'), row=row, col=1)
        
        # Total line
        total = norm + red + rej
        fig.add_trace(go.Scatter(x=x, y=total, name="Total Activity", line=dict(color='black', width=1)), row=row, col=1)
        
    fig.update_layout(barmode='stack')
    apply_standard_layout(fig, "Trade Flow & Circuit Breaker Activity")
    return fig

def make_tradeflow_panel(model: SingleModelResults) -> go.Figure:
    return go.Figure()
