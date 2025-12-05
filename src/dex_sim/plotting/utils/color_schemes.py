import plotly.colors as pc

# Breaker State Colors
BREAKER_COLORS = {
    0: "rgba(240, 240, 240, 0.5)",   # Normal - Light Gray
    1: "rgba(255, 165, 0, 0.3)",     # Soft - Orange/Yellow
    2: "rgba(255, 0, 0, 0.3)"        # Hard - Red
}

# Model Colors (Cyclic)
MODEL_PALETTE = pc.qualitative.Plotly

def get_model_color(idx: int) -> str:
    return MODEL_PALETTE[idx % len(MODEL_PALETTE)]

# Trace Colors
TRACE_COLORS = {
    "long": "#2ca02c",  # Green
    "short": "#d62728", # Red
    "price": "#1f77b4", # Blue
    "im": "#9467bd",    # Purple
    "mm": "#8c564b",    # Brown
    "liq": "#e377c2",   # Pink
}
