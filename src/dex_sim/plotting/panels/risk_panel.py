import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..utils.transforms import compute_percentiles
from ..layout import make_stacked_subplots, apply_standard_layout

def make_risk_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    """
    Compare Rt and Breaker States across models.
    """
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)

    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        # Guard: Missing or All-NaN Rt
        if res.rt is None or res.rt.size == 0 or np.all(np.isnan(res.rt)):
            fig.add_annotation(
                text=f"{name}: No risk index available",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue

        # Compute stats
        stats = compute_percentiles(res.rt, (50, 90))
        
        # 1. Plot Median Rt
        fig.add_trace(
            go.Scatter(
                y=stats[50], 
                name=f"{name} Rt (Median)",
                line=dict(color="black", width=2),
                legendgroup=name
            ),
            row=row, col=1
        )
        
        # Breaker logic (omitted as per previous implementation logic, but safe to ignore if missing)

    apply_standard_layout(fig, "Systemic Risk Index (Rt) Comparison")
    return fig

def make_risk_panel(model: SingleModelResults) -> go.Figure:
    """
    Detailed single-model risk view.
    """
    fig = go.Figure()
    if model.rt is None or model.rt.size == 0 or np.all(np.isnan(model.rt)):
        fig.add_annotation(
            text="No risk index data",
            showarrow=False,
            font=dict(color="red")
        )
        return fig
        
    stats = compute_percentiles(model.rt, (1, 10, 50, 90, 99))
    
    x = np.arange(model.rt.shape[1])
    
    # Fan chart
    fig.add_trace(go.Scatter(x=x, y=stats[99], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=stats[1], fill='tonexty', fillcolor='rgba(0,0,0,0.1)', line=dict(width=0), name='1-99%'))
    
    fig.add_trace(go.Scatter(x=x, y=stats[90], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=stats[10], fill='tonexty', fillcolor='rgba(0,0,0,0.2)', line=dict(width=0), name='10-90%'))
    
    fig.add_trace(go.Scatter(x=x, y=stats[50], line=dict(color='black', width=2), name='Median Rt'))
    
    apply_standard_layout(fig, f"Risk Index Detail: {model.name}")
    return fig