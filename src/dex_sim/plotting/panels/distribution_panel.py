import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...data_structures import SingleModelResults
from ..layout import apply_standard_layout

def make_distribution_compare_panels(models: dict[str, SingleModelResults]) -> go.Figure:
    
    rows = len(models)
    titles = list(models.keys())
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles)
    
    for idx, (name, res) in enumerate(models.items()):
        row = idx + 1
        
        has_df = res.df_required is not None and res.df_required.size > 0 and not np.all(np.isnan(res.df_required))
        
        if not has_df:
            fig.add_annotation(
                text=f"{name}: No DF data",
                row=row, col=1,
                showarrow=False,
                font=dict(color="red")
            )
            continue
        
        fig.add_trace(
            go.Histogram(x=res.df_required, name=f"{name} DF Req", opacity=0.7),
            row=row, col=1
        )
            
    apply_standard_layout(fig, "Default Fund Requirement Distribution")
    return fig

def make_distribution_panel(model: SingleModelResults) -> go.Figure:
    return go.Figure()