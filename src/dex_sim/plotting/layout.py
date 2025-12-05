import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_stacked_subplots(
    num_rows: int, 
    row_titles: list[str],
    shared_x: bool = True,
    vertical_spacing: float = 0.05,
    height_per_row: int = 300
) -> go.Figure:
    """
    Creates a vertical stack of subplots, one per model.
    """
    fig = make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=shared_x,
        vertical_spacing=vertical_spacing,
        subplot_titles=row_titles
    )
    fig.update_layout(height=num_rows * height_per_row, showlegend=True)
    return fig

def apply_standard_layout(fig: go.Figure, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified"
    )
