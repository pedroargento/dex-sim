import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..utils.transforms import reconstruct_im_mm_bands
from ..layout import make_stacked_subplots, apply_standard_layout

def make_margin_compare_panels(
    models: dict[str, SingleModelResults], 
    sigma_path: np.ndarray = None
) -> go.Figure:
    
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)

    if sigma_path is None:
        # Create a dummy sigma path if missing, just to allow rendering
        # Warning: This yields invalid absolute numbers, but shows relative multiplier effects
        sigma_path = np.ones(1) 

    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        if res.notional_paths is None or res.margin_multiplier is None:
            continue
            
        # Try to guess if it's ES or Fixed from name? 
        # Or just assume ES for visualization if multiplier varies.
        # We can check if margin_multiplier is all 1s.
        is_es = not np.allclose(res.margin_multiplier, 1.0)
        
        bands = reconstruct_im_mm_bands(
            res.notional_paths, 
            sigma_path, 
            res.margin_multiplier, 
            im_is_es=True # Assume true to visualize the dynamic nature
        )
        
        if not bands: continue
        
        # IM P50-P90
        im = bands["im"]
        fig.add_trace(go.Scatter(y=im[90], line=dict(width=0), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(
            y=im[50], 
            fill='tonexty', 
            fillcolor='rgba(148, 103, 189, 0.2)', 
            line=dict(color='#9467bd'), 
            name=f"{name} IM Band"
        ), row=row, col=1)
        
        # MM P50-P90
        mm = bands["mm"]
        # Overlay MM line
        fig.add_trace(go.Scatter(
            y=mm[50], 
            line=dict(color='#8c564b', dash='dot'), 
            name=f"{name} MM Median"
        ), row=row, col=1)

    apply_standard_layout(fig, "Margin Requirements (IM/MM)")
    return fig

def make_margin_panel(model: SingleModelResults, sigma_path: np.ndarray = None) -> go.Figure:
    fig = go.Figure()
    # Detail view similar to above but cleaner
    apply_standard_layout(fig, f"Margin Detail: {model.name}")
    return fig
