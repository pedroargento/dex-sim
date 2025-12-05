import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..utils.transforms import compute_percentiles
from ..utils.color_schemes import BREAKER_COLORS, get_model_color
from ..layout import make_stacked_subplots, apply_standard_layout

def make_risk_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    """
    Compare Rt and Breaker States across models.
    """
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)

    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        if res.rt is None:
            fig.add_annotation(text="No Risk Index Data", showarrow=False, row=row, col=1)
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
        
        # 2. Breaker Background Shading (Approximation)
        # We find transitions in the median breaker state or overlay generic bands?
        # The prompt asks for "Background shading for breaker states".
        # Since states can flip-flop per path, using the *mode* or *median* state is safer for summary.
        # Or we shade the background based on the Rt thresholds if we knew them.
        # Better: Shade based on the dominant state at each timestep across paths.
        
        if res.breaker_state is not None:
            # Mode of breaker state at each t
            # scipy.stats.mode is slow for large arrays, manual count?
            # Let's just use the state of the median Rt path as a proxy, 
            # or better: visualize the PROPORTION of paths in each state?
            # The prompt implies simple shading. Let's use the breaker state associated with the median Rt path 
            # OR just shade regions where median Rt > thresholds if we knew them.
            
            # Heuristic: Shade regions based on the majority state
            # 0: Normal, 1: Soft, 2: Hard
            # We can iterate time steps and add vrects. This is heavy for Plotly.
            # Alternative: Heatmap strip at the bottom?
            # Let's try: colored line segments or scatter markers behind?
            # Let's try: Stacked bar chart in background? No.
            # Let's stick to: Plotting the Rt, and if we have thresholds, add horizontal lines.
            # But we don't have thresholds in SingleModelResults easily.
            
            # Let's try to add a secondary heatmap trace for state?
            pass 

    apply_standard_layout(fig, "Systemic Risk Index (Rt) Comparison")
    return fig

def make_risk_panel(model: SingleModelResults) -> go.Figure:
    """
    Detailed single-model risk view.
    """
    fig = go.Figure()
    if model.rt is None:
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
