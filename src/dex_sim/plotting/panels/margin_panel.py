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

    if sigma_path is None or sigma_path.size == 0:
        # Create a dummy sigma path if missing, just to allow rendering
        sigma_path = np.ones(1) 

    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        # Guards
        has_notional = res.notional_paths is not None and res.notional_paths.size > 0
        has_mult = res.margin_multiplier is not None and res.margin_multiplier.size > 0 and not np.all(np.isnan(res.margin_multiplier))
        
        if not has_notional or not has_mult:
            fig.add_annotation(
                text=f"{name}: No margin data",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue
            
        bands = reconstruct_im_mm_bands(
            res.notional_paths, 
            sigma_path, 
            res.margin_multiplier, 
            im_is_es=True 
        )
        
        if not bands: 
            fig.add_annotation(
                text=f"{name}: Failed to reconstruct bands",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue
        
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
    apply_standard_layout(fig, f"Margin Detail: {model.name}")
    return fig