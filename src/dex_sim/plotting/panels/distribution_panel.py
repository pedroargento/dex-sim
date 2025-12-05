import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...data_structures import SingleModelResults
from ..layout import apply_standard_layout

def make_distribution_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    # 1 row, N cols (side by side histograms) or N rows?
    # "1 subplot per model".
    # For distributions, vertical stack is good for comparing shape.
    
    rows = len(models)
    titles = list(models.keys())
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        # Default Fund Required Distribution
        if res.df_required is not None:
            fig.add_trace(
                go.Histogram(x=res.df_required, name=f"{name} DF Req", opacity=0.7),
                row=row, col=1
            )
            
    apply_standard_layout(fig, "Default Fund Requirement Distribution")
    return fig

def make_distribution_panel(model: SingleModelResults) -> go.Figure:
    return go.Figure()
