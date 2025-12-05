import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..layout import make_stacked_subplots, apply_standard_layout

def make_tradeflow_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        # Guards
        has_norm = res.intent_accepted_normal is not None and res.intent_accepted_normal.size > 0
        has_red = res.intent_accepted_reduce is not None and res.intent_accepted_reduce.size > 0
        has_rej = res.intent_rejected is not None and res.intent_rejected.size > 0
        
        if not (has_norm or has_red or has_rej):
            fig.add_annotation(
                text=f"{name}: No tradeflow data",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue
        
        # Compute mean across paths
        # We handle missing arrays by treating them as zeros if AT LEAST ONE exists
        # But per requirements, if mostly missing, maybe just skip?
        # Let's be robust: if one exists, we plot.
        
        def safe_mean(arr):
            if arr is None or arr.size == 0 or np.all(np.isnan(arr)):
                # Return zeros of correct shape if possible? 
                # We need T dimension. 
                # If we don't know T, we can't create zeros.
                return None
            return np.nanmean(arr, axis=0)
            
        norm = safe_mean(res.intent_accepted_normal)
        red = safe_mean(res.intent_accepted_reduce)
        rej = safe_mean(res.intent_rejected)
        
        # Determine T from whatever is valid
        T = 0
        if norm is not None: T = len(norm)
        elif red is not None: T = len(red)
        elif rej is not None: T = len(rej)
        
        if T == 0:
             fig.add_annotation(text=f"{name}: No valid tradeflow data", row=row, col=1, showarrow=False, font=dict(color="red"))
             continue

        x = np.arange(T)
        
        if norm is None: norm = np.zeros(T)
        if red is None: red = np.zeros(T)
        if rej is None: rej = np.zeros(T)
        
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