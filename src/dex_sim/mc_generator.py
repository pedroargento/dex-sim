# risk_sim/mc_generator.py
import json
from typing import Optional

import numpy as np
from scipy.stats import t


class MCReturnsGenerator:
    """
    ETH-like Monte Carlo generator with t-distributed shocks and GARCH-ish vol.
    """

    def __init__(
        self,
        garch_params_file: str = "garch_params.json",
        num_paths: int = 10_000,
        horizon: int = 7_200,  # 5 days * 1440 minutes/day
        df: int = 6,
        shock_corr: float = -0.7,
        stress_factor: float = 1.0,
    ):
        with open(garch_params_file, "r") as f:
            params = json.load(f)

        self.omega = params["omega"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.last_sigma_t = params["last_sigma_t"]  # daily vol

        self.num_paths = num_paths
        self.horizon = horizon
        self.df = df
        self.stress_factor = stress_factor

        corr = np.array([[1.0, shock_corr], [shock_corr, 1.0]])
        self.cholesky = np.linalg.cholesky(corr)

        # daily → minute
        self.TIME_SCALE = np.sqrt(1.0 / 1440.0)

    def generate_paths(self, num_paths: Optional[int] = None):
        if num_paths is None:
            num_paths = self.num_paths

        t_indep = np.random.standard_t(self.df, size=(num_paths, self.horizon, 2))
        shocks = t_indep @ self.cholesky.T
        shocks *= self.stress_factor

        price_shocks = shocks[:, :, 0]
        liquidity_shocks = shocks[:, :, 1]

        log_returns = np.zeros((num_paths, self.horizon))
        sigmas = np.zeros((num_paths, self.horizon))
        amihud_le = np.zeros((num_paths, self.horizon))

        sigmas[:, 0] = self.last_sigma_t

        for t_step in range(1, self.horizon):
            sigma_sq = (
                self.omega
                + self.alpha * log_returns[:, t_step - 1] ** 2
                + self.beta * sigmas[:, t_step - 1] ** 2
            )
            sigma = np.sqrt(np.maximum(sigma_sq, 0.0))
            sigma = np.clip(sigma, 0.03, 0.20)  # sanity bounds
            sigmas[:, t_step] = sigma

            sigma_step = sigma * self.TIME_SCALE
            log_returns[:, t_step] = sigma_step * price_shocks[:, t_step]

            raw = 1.0 / (np.abs(liquidity_shocks[:, t_step]) + 1e-3)
            amihud_le[:, t_step] = np.clip(np.log1p(raw), 0.0, 10.0)

        log_returns = np.clip(log_returns, -0.15, 0.15)
        return log_returns, amihud_le, sigmas
