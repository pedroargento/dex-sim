import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...data_structures import SingleModelResults
from ..layout import apply_standard_layout

def make_path_drilldown_panel(
    models: dict[str, SingleModelResults], 
    num_paths_to_expose: int = 10
) -> go.Figure:
    """
    Creates an interactive figure to drill down into specific paths.
    Uses Plotly updatemenus to switch between (Model, Path) combinations.
    
    Note: Due to browser limits, we only expose a subset of paths.
    """
    
    # Layout: 4 rows
    # 1. Price + Events
    # 2. Rt + Breaker
    # 3. Leverage
    # 4. Equity / Position
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=["Price & Events", "Risk Index (Rt)", "Leverage", "Equity & Position"]
    )
    
    # We need to generate traces for every combination of (Model, Path)
    # keeping them invisible initially, except the first one.
    
    traces = []
    visibility_masks = []
    dropdown_buttons = []
    
    trace_cursor = 0
    
    path_indices = range(min(num_paths_to_expose, list(models.values())[0].price_paths.shape[0]))
    
    for model_name, res in models.items():
        for p_idx in path_indices:
            
            # --- 1. Price & Events ---
            
            # Price Line
            fig.add_trace(go.Scatter(y=res.price_paths[p_idx], name="Price", line=dict(color='blue'), visible=False), row=1, col=1)
            
            # Liquidation Markers
            if res.liquidation_fraction is not None:
                liqs = res.liquidation_fraction[p_idx] > 0
                if np.any(liqs):
                    x_liq = np.where(liqs)[0]
                    y_liq = res.price_paths[p_idx][liqs]
                    fig.add_trace(go.Scatter(
                        x=x_liq, y=y_liq, mode='markers', 
                        marker=dict(symbol='x', color='red', size=10),
                        name="Liquidation", visible=False
                    ), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=[], y=[], name="Liquidation", visible=False), row=1, col=1) # Placeholder
            else:
                fig.add_trace(go.Scatter(x=[], y=[], visible=False), row=1, col=1)

            # --- 2. Rt ---
            if res.rt is not None:
                fig.add_trace(go.Scatter(y=res.rt[p_idx], name="Rt", line=dict(color='black'), visible=False), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=2, col=1)

            # --- 3. Leverage ---
            if res.lev_long is not None:
                fig.add_trace(go.Scatter(y=res.lev_long[p_idx], name="Lev Long", line=dict(color='green'), visible=False), row=3, col=1)
                fig.add_trace(go.Scatter(y=res.lev_short[p_idx], name="Lev Short", line=dict(color='red'), visible=False), row=3, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=3, col=1)
                fig.add_trace(go.Scatter(y=[], visible=False), row=3, col=1)

            # --- 4. Position / Equity ---
            # Let's plot Net Position (Notional / Price)
            if res.notional_paths is not None:
                # Approx position
                pos = res.notional_paths[p_idx] / res.price_paths[p_idx]
                fig.add_trace(go.Scatter(y=pos, name="Net Pos", line=dict(color='purple'), visible=False), row=4, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=4, col=1)

            # Calculate how many traces we just added
            # 1 (Price) + 1 (Liq) + 1 (Rt) + 2 (Lev) + 1 (Pos) = 6 traces per combo
            # Ensure this count is constant!
            
            num_traces_per_combo = 6
            
            # Create visibility mask
            # It should be all False, except the range [trace_cursor, trace_cursor + 6] is True
            
            # We'll construct the mask dynamically in the button
            # But Plotly needs the full boolean array or "restyle" command indices.
            # "restyle" with 'visible' and list of booleans is standard.
            # Wait, generating specific masks for 100s of buttons is heavy.
            # But for < 50 it's fine.
            
            label = f"{model_name} - Path {p_idx}"
            
            # We need to know the total number of traces in the FINAL figure to build the mask correctly.
            # This requires two passes or simple math: Total Traces = (Models * Paths * TracesPerCombo)
            
            dropdown_buttons.append(dict(
                label=label,
                method="update",
                args=[{"visible": [False] * len(models) * len(path_indices) * num_traces_per_combo}, # Placeholder
                      {"title": label}]
            ))
            
            # Fix the mask later
            dropdown_buttons[-1]['args'][0]['visible'][trace_cursor : trace_cursor + num_traces_per_combo] = [True] * num_traces_per_combo
            
            trace_cursor += num_traces_per_combo

    # Set initial visibility
    if fig.data:
        for i in range(6):
            fig.data[i].visible = True

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=1.15, y=1,
            xanchor='left', yanchor='top'
        )]
    )
    
    apply_standard_layout(fig, "Path Drilldown")
    return fig
