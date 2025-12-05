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
    """
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=["Price & Events", "Risk Index (Rt)", "Leverage", "Equity & Position"]
    )
    
    dropdown_buttons = []
    trace_cursor = 0
    
    # Safely get path indices
    first_res = list(models.values())[0]
    if first_res.price_paths is not None:
        max_paths = first_res.price_paths.shape[0]
    else:
        max_paths = 0
    
    path_indices = range(min(num_paths_to_expose, max_paths))
    
    # Helper for valid data check
    def has_data(arr):
        return arr is not None and arr.size > 0 and not np.all(np.isnan(arr))

    for model_name, res in models.items():
        for p_idx in path_indices:
            
            # --- 1. Price & Events ---
            if has_data(res.price_paths):
                fig.add_trace(go.Scatter(y=res.price_paths[p_idx], name="Price", line=dict(color='blue'), visible=False), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], name="Price", visible=False), row=1, col=1)
            
            # Liquidation Markers
            if has_data(res.liquidation_fraction):
                # Safe check for > 0 handling NaNs
                liqs = np.nan_to_num(res.liquidation_fraction[p_idx], nan=0.0) > 0
                if np.any(liqs) and has_data(res.price_paths):
                    x_liq = np.where(liqs)[0]
                    y_liq = res.price_paths[p_idx][liqs]
                    fig.add_trace(go.Scatter(
                        x=x_liq, y=y_liq, mode='markers', 
                        marker=dict(symbol='x', color='red', size=10),
                        name="Liquidation", visible=False
                    ), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=[], y=[], name="Liquidation", visible=False), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=[], y=[], visible=False), row=1, col=1)

            # --- 2. Rt ---
            if has_data(res.rt):
                fig.add_trace(go.Scatter(y=res.rt[p_idx], name="Rt", line=dict(color='black'), visible=False), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=2, col=1)

            # --- 3. Leverage ---
            # Long
            if has_data(res.lev_long):
                fig.add_trace(go.Scatter(y=res.lev_long[p_idx], name="Lev Long", line=dict(color='green'), visible=False), row=3, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=3, col=1)
                
            # Short
            if has_data(res.lev_short):
                fig.add_trace(go.Scatter(y=res.lev_short[p_idx], name="Lev Short", line=dict(color='red'), visible=False), row=3, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=3, col=1)

            # --- 4. Position / Equity ---
            if has_data(res.notional_paths) and has_data(res.price_paths):
                pos = res.notional_paths[p_idx] / res.price_paths[p_idx]
                fig.add_trace(go.Scatter(y=pos, name="Net Pos", line=dict(color='purple'), visible=False), row=4, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], visible=False), row=4, col=1)

            # 6 traces per combo
            num_traces_per_combo = 6
            
            label = f"{model_name} - Path {p_idx}"
            
            # Placeholder mask
            mask = [False] * (len(models) * len(path_indices) * num_traces_per_combo)
            # Set this combo to True
            # (Note: this manual list building is inefficient for large N, but ok for small drilldowns)
            # Efficient way: create mask of all False, then slice assign
            
            # We can't actually modify the button arg AFTER creation easily in the loop if we append.
            # So we create the mask here.
            # But wait, we need total length.
            # The loop is deterministic. Total len = len(models) * len(path_indices) * 6
            
            dropdown_buttons.append(dict(
                label=label,
                method="update",
                args=[{"visible": [False] * len(models) * len(path_indices) * num_traces_per_combo}, 
                      {"title": label}]
            ))
            
            # Now set the slice for this specific button to True
            dropdown_buttons[-1]['args'][0]['visible'][trace_cursor : trace_cursor + num_traces_per_combo] = [True] * num_traces_per_combo
            
            trace_cursor += num_traces_per_combo

    # Set initial visibility (first combo)
    if fig.data:
        for i in range(min(6, len(fig.data))):
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