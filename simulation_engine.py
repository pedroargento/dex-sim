
import json
import numpy as np
from scipy.stats import t

class MCReturnsGenerator:
    """
    Generates Monte Carlo price paths using a GARCH(1,1) model with T-distributed shocks.
    """
    def __init__(self, garch_params_file='garch_params.json', num_paths=10000, horizon=7200, df=4):
        with open(garch_params_file, 'r') as f:
            params = json.load(f)
        self.omega = params['omega']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.last_sigma_t = params['last_sigma_t']
        self.num_paths = num_paths
        self.horizon = horizon  # e.g., 5 days of minute data = 5 * 24 * 60 = 7200
        self.df = df
        # Inverse correlation between price shocks and liquidity proxy
        self.shock_corr = -0.7

    def generate_single_path(self):
        """
        Generates a single correlated price and liquidity shock path.
        This method uses a Cholesky decomposition to induce correlation between two
        independent T-distributed series, which is much faster than using the inverse CDF.
        """
        # Time scaling factor for converting daily sigma to per-minute sigma
        TIME_SCALE = np.sqrt(1.0 / 1440.0)

        # Generate independent T-distributed random variables
        t_shocks_independent = np.random.standard_t(self.df, size=(self.horizon, 2))

        # Create correlation matrix and compute Cholesky decomposition
        corr_matrix = np.array([[1.0, self.shock_corr], [self.shock_corr, 1.0]])
        cholesky_decomp = np.linalg.cholesky(corr_matrix)

        # Induce correlation
        t_shocks_correlated = t_shocks_independent @ cholesky_decomp.T
        
        price_shocks = t_shocks_correlated[:, 0]
        liquidity_shocks = t_shocks_correlated[:, 1]

        log_returns = np.zeros(self.horizon)
        sigmas = np.zeros(self.horizon)
        amihud_le = np.zeros(self.horizon)

        sigmas[0] = self.last_sigma_t

        for t_step in range(1, self.horizon):
            # Update GARCH sigma (daily)
            sigma_daily = np.sqrt(self.omega + self.alpha * (log_returns[t_step - 1]**2) + self.beta * (sigmas[t_step - 1]**2))
            
            # Clamp daily sigma to prevent volatility explosion
            sigmas[t_step] = np.clip(sigma_daily, 0.01, 0.25)
            
            # Convert to per-minute sigma for log return calculation
            sigma_step = sigmas[t_step] * TIME_SCALE
            
            log_returns[t_step] = sigma_step * price_shocks[t_step]

            # Fix Amihud liquidity proxy
            raw = 1.0 / (np.abs(liquidity_shocks[t_step]) + 1e-3)
            amihud_le[t_step] = np.clip(np.log1p(raw), 0.0, 10.0)

        # Clamp log returns to prevent exp overflow
        log_returns = np.clip(log_returns, -0.3, 0.3)

        return log_returns, amihud_le

class TraderAgent:
    """
    Represents a single trader in the simulation.
    """
    def __init__(self, position, initial_margin, equity):
        self.position = position  # Number of contracts
        self.initial_margin = initial_margin
        self.equity = equity

class RiskEngine:
    """
    Calculates initial margin (IM) based on Parametric Expected Shortfall (ES).
    """
    def __init__(self, confidence_level=0.99, df=4):
        self.confidence_level = confidence_level
        self.df = df

    def calculate_im(self, sigma_t, position_notional):
        """
        Calculates Initial Margin using Parametric ES for a T-distribution.
        """
        t_inv = t.ppf(self.confidence_level, self.df)
        es_factor = (t.pdf(t_inv, self.df) / (1 - self.confidence_level)) * ((self.df + t_inv**2) / (self.df - 1))
        
        expected_shortfall = sigma_t * es_factor
        
        # IM is the expected shortfall of the position
        im = expected_shortfall * position_notional
        return im

class SystemAggregator:
    """
    Calculates the Systemic Stress Index (R_t) and determines the circuit breaker state.
    """
    def __init__(self, weights=None, thresholds=None):
        if weights is None:
            # Placeholder weights for the R_t formula
            self.weights = {'delta': 0.2, 'gamma': 0.1, 'var': 0.4, 'le': 0.3}
        else:
            self.weights = weights
            
        if thresholds is None:
            # Placeholder thresholds for circuit breaker states
            self.thresholds = {'soft': 1.5, 'hard': 2.5}
        else:
            self.thresholds = thresholds
            
    def calculate_rt(self, sys_delta, sys_gamma, sys_var, le_t):
        """
        Calculates the Systemic Stress Index R_t.
        """
        rt = (self.weights['delta'] * np.abs(sys_delta) +
              self.weights['gamma'] * sys_gamma +
              self.weights['var'] * sys_var +
              self.weights['le'] * le_t)
        return rt

    def get_circuit_breaker_state(self, rt):
        """
        Determines the circuit breaker state based on R_t.
        """
        if rt < self.thresholds['soft']:
            return "NORMAL", 1.0  # Margin multiplier
        elif self.thresholds['soft'] <= rt < self.thresholds['hard']:
            return "SOFT", 1.5
        else:
            return "HARD", 2.0
