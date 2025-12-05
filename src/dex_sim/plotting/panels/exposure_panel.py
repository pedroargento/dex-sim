import numpy as np
import plotly.graph_objects as go
from ...data_structures import SingleModelResults
from ..layout import make_stacked_subplots, apply_standard_layout

def make_exposure_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    model_names = list(models.keys())
    fig = make_stacked_subplots(len(model_names), model_names)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        if res.notional_paths is None: continue
        
        # Total Open Interest (Mean across paths)
        # notional_paths in data_structures seems to be aggregate?
        # "notional_paths: [P, T]"
        # If it's aggregate system OI, we just plot mean.
        
        oi = np.nanmean(res.notional_paths, axis=0)
        
        x = np.arange(len(oi))
        fig.add_trace(go.Scatter(x=x, y=oi, name=f"{name} OI", fill='tozeroy'), row=row, col=1)
        
        if res.ecp_position_path is not None:
             ecp = np.nanmean(res.ecp_position_path, axis=0)
             fig.add_trace(go.Scatter(x=x, y=ecp, name=f"{name} ECP Net", line=dict(color='black', dash='dot')), row=row, col=1)

    apply_standard_layout(fig, "Systemic Exposure")
    return fig

def make_exposure_panel(model: SingleModelResults) -> go.Figure:
    return go.Figure()
