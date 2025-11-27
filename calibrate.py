import pandas as pd
import numpy as np
from arch import arch_model
import json

def calibrate_garch():
    """
    Loads price data, calculates log returns and Amihud Illiquidity,
    calibrates a GARCH(1,1) model, and saves the parameters.
    """
    # Load data
    df = pd.read_csv('data/ethusdt_1m2024-2025.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Calculate log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()

    # Calculate Amihud Illiquidity Proxy (LE_t)
    # Replace 0s in quote_volume to avoid division by zero errors
    df['quote_volume'] = df['quote_volume'].replace(0, np.nan).fillna(method='ffill')
    df['amihud_liquidity'] = df['log_returns'].abs() / df['quote_volume']
    
    # Calibrate GARCH(1,1) model, rescaling returns to improve convergence
    rescaled_returns = df['log_returns'] * 1000
    garch_model = arch_model(rescaled_returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off', options={'ftol': 1e-10})

    # Extract parameters and scale them back
    omega = garch_fit.params['omega'] / (1000**2)
    alpha = garch_fit.params['alpha[1]']
    beta = garch_fit.params['beta[1]']
    
    # Get the last observed EWMA volatility and scale it back
    last_sigma_t = garch_fit.conditional_volatility[-1] / 1000

    # Save parameters to a JSON file
    garch_params = {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'last_sigma_t': last_sigma_t
    }

    with open('garch_params.json', 'w') as f:
        json.dump(garch_params, f, indent=4)

    print("GARCH parameters saved to garch_params.json")
    print(garch_fit.summary())

if __name__ == "__main__":
    calibrate_garch()