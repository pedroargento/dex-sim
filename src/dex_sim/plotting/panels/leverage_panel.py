import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..utils.transforms import compute_percentiles
from ..layout import make_stacked_subplots, apply_standard_layout

def make_leverage_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)
    
    # Determine global y-max for scale consistency
    global_max = 0
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        has_long = res.lev_long is not None and res.lev_long.size > 0 and not np.all(np.isnan(res.lev_long))
        has_short = res.lev_short is not None and res.lev_short.size > 0 and not np.all(np.isnan(res.lev_short))
        
        if not has_long and not has_short:
            fig.add_annotation(
                text=f"{name}: No leverage data",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue
        
        # Long Lev
        if has_long:
            l_stats = compute_percentiles(res.lev_long, (50, 95, 99))
            fig.add_trace(go.Scatter(y=l_stats[50], name=f"{name} Long P50", line=dict(width=2)), row=row, col=1)
            fig.add_trace(go.Scatter(y=l_stats[95], name=f"{name} Long P95", line=dict(dash='dot')), row=row, col=1)
            if not np.all(np.isnan(l_stats[99])):
                global_max = max(global_max, np.nanmax(l_stats[99]))
            
        # Short Lev
        if has_short:
            s_stats = compute_percentiles(res.lev_short, (50, 95, 99))
            fig.add_trace(go.Scatter(y=s_stats[50], name=f"{name} Short P50", line=dict(width=2)), row=row, col=1)
    
    # Safely set y-range
    if global_max > 0:
        fig.update_yaxes(range=[0, min(global_max * 1.1, 100)]) 
        
    apply_standard_layout(fig, "Leverage Comparison")
    return fig

def make_leverage_panel(model: SingleModelResults) -> go.Figure:
    fig = go.Figure()
    
    has_long = model.lev_long is not None and model.lev_long.size > 0 and not np.all(np.isnan(model.lev_long))
    
    if not has_long:
        fig.add_annotation(text="No leverage data", showarrow=False, font=dict(color="red"))
        return fig
    
    l_stats = compute_percentiles(model.lev_long, (50, 90, 99))
    
    x = np.arange(l_stats[50].shape[0])
    
    fig.add_trace(go.Scatter(x=x, y=l_stats[50], name="Long P50"))
    fig.add_trace(go.Scatter(x=x, y=l_stats[99], name="Long P99"))
    
    apply_standard_layout(fig, f"Leverage Detail: {model.name}")
    return fig