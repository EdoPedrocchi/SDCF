"""
RevenueModeler: Fits three state-space / time-series models to log-revenue
and selects the best one via Likelihood Ratio Tests (Bottazzi et al., §3).

Models
------
1. AR(1)               – on first-differences of log-revenue
2. Local Level         – random walk + noise  (Kalman Filter)
3. Local Linear Trend  – random walk + slope + noise (Kalman Filter)

LRT hierarchy (chi-sq, df=1 at each step):
    Step 1: AR(1) vs Local Level
    Step 2: Local Level vs Local Linear Trend
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ModelName = Literal["AR1", "LocalLevel", "LocalLinearTrend"]
LRT_ALPHA = 0.05


@dataclass
class FittedModel:
    """Stores the fitted model and its parameters for Monte Carlo simulation."""
    name: ModelName
    log_likelihood: float

    # AR(1) params
    ar1_phi:   float = 0.0
    ar1_sigma: float = 0.0
    ar1_mu:    float = 0.0

    # State-space params
    sigma_level: float = 0.0
    sigma_slope: float = 0.0
    sigma_obs:   float = 0.0

    # Last Kalman-filtered state (seeds the simulation)
    last_level:       float = 0.0
    last_slope:       float = 0.0
    last_log_revenue: float = 0.0

    # LRT metadata (populated by fit_and_select)
    lrt_statistic_ar1_vs_ll:  float = 0.0
    lrt_p_value_ar1_vs_ll:    float = 1.0
    lrt_statistic_ll_vs_llt:  float = 0.0
    lrt_p_value_ll_vs_llt:    float = 1.0
    all_log_likelihoods: dict = field(default_factory=dict)


class RevenueModeler:
    """
    Fits revenue models and selects the best via LRT.

    Parameters
    ----------
    revenue : pd.Series
        Annual revenue levels (not logs).
    min_obs : int
        Minimum observations required (default 5).
    """

    def __init__(self, revenue: pd.Series, min_obs: int = 5):
        self.revenue = revenue.astype(float)
        if len(self.revenue) < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations; got {len(self.revenue)}."
            )
        self.log_rev      = np.log(self.revenue.values.astype(float))
        self.diff_log_rev = np.diff(self.log_rev)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_and_select(self) -> FittedModel:
        """Fit all three models and select the best via LRT."""
        ar1 = self._fit_ar1()
        ll  = self._fit_local_level()
        llt = self._fit_local_linear_trend()

        # Step 1: AR(1) vs Local Level
        stat1  = max(0.0, 2.0 * (ll.log_likelihood  - ar1.log_likelihood))
        pval1  = 1.0 - stats.chi2.cdf(stat1, df=1)

        # Step 2: Local Level vs LLT
        stat2  = max(0.0, 2.0 * (llt.log_likelihood - ll.log_likelihood))
        pval2  = 1.0 - stats.chi2.cdf(stat2, df=1)

        logger.info(
            f"LRT AR1→LL:  stat={stat1:.4f}, p={pval1:.4f} | "
            f"LRT LL→LLT: stat={stat2:.4f}, p={pval2:.4f}"
        )

        # Select: move to more complex model only when LRT is significant
        if pval1 > LRT_ALPHA:
            selected = ar1
        elif pval2 > LRT_ALPHA:
            selected = ll
        else:
            selected = llt

        selected.lrt_statistic_ar1_vs_ll  = stat1
        selected.lrt_p_value_ar1_vs_ll    = pval1
        selected.lrt_statistic_ll_vs_llt  = stat2
        selected.lrt_p_value_ll_vs_llt    = pval2
        selected.all_log_likelihoods = {
            "AR1":              ar1.log_likelihood,
            "LocalLevel":       ll.log_likelihood,
            "LocalLinearTrend": llt.log_likelihood,
        }

        logger.info(f"Selected model: {selected.name}")
        return selected

    # ------------------------------------------------------------------
    # Model 1: AR(1) on first differences
    # ------------------------------------------------------------------

    def _fit_ar1(self) -> FittedModel:
        """Δlog(R_t) = μ + φ·Δlog(R_{t-1}) + ε_t via OLS."""
        y = self.diff_log_rev
        if len(y) < 3:
            raise ValueError("Need ≥ 3 first-differenced obs for AR(1).")

        ols   = OLS(y[1:], add_constant(y[:-1])).fit()
        mu    = float(ols.params[0])
        phi   = float(ols.params[1])
        resid = ols.resid
        sigma = max(float(np.std(resid, ddof=2)), 1e-8)

        n       = len(resid)
        log_lik = (
            -0.5 * n * np.log(2.0 * np.pi * sigma ** 2)
            - np.sum(resid ** 2) / (2.0 * sigma ** 2)
        )

        return FittedModel(
            name="AR1",
            log_likelihood=float(log_lik),
            ar1_phi=phi,
            ar1_sigma=sigma,
            ar1_mu=mu,
            last_log_revenue=self.log_rev[-1],
        )

    # ------------------------------------------------------------------
    # Model 2: Local Level
    # ------------------------------------------------------------------

    def _fit_local_level(self) -> FittedModel:
        """
        y_t  = μ_t + ε_t,     ε_t ~ N(0, σ²_ε)
        μ_t  = μ_{t-1} + η_t, η_t ~ N(0, σ²_η)
        """
        mod    = UnobservedComponents(self.log_rev, level="local level")
        result = self._fit_ssm(mod, "LocalLevel")
        # statsmodels param_names: ['sigma2.irregular', 'sigma2.level']
        sigma_obs   = np.sqrt(abs(self._get_param(result, "sigma2.irregular", 0)))
        sigma_level = np.sqrt(abs(self._get_param(result, "sigma2.level",     1)))
        last_level  = float(result.smoother_results.smoothed_state[0, -1])

        return FittedModel(
            name="LocalLevel",
            log_likelihood=float(result.llf),
            sigma_obs=float(sigma_obs),
            sigma_level=float(sigma_level),
            last_level=last_level,
            last_log_revenue=self.log_rev[-1],
        )

    # ------------------------------------------------------------------
    # Model 3: Local Linear Trend
    # ------------------------------------------------------------------

    def _fit_local_linear_trend(self) -> FittedModel:
        """
        y_t  = μ_t + ε_t,               ε_t ~ N(0, σ²_ε)
        μ_t  = μ_{t-1} + ν_{t-1} + η_t, η_t ~ N(0, σ²_η)
        ν_t  = ν_{t-1} + ζ_t,            ζ_t ~ N(0, σ²_ζ)
        """
        mod    = UnobservedComponents(self.log_rev, level="local linear trend")
        result = self._fit_ssm(mod, "LocalLinearTrend")
        # statsmodels param_names: ['sigma2.irregular', 'sigma2.level', 'sigma2.trend']
        sigma_obs   = np.sqrt(abs(self._get_param(result, "sigma2.irregular", 0)))
        sigma_level = np.sqrt(abs(self._get_param(result, "sigma2.level",     1)))
        sigma_slope = np.sqrt(abs(self._get_param(result, "sigma2.trend",     2)))

        sm         = result.smoother_results
        last_level = float(sm.smoothed_state[0, -1])
        last_slope = float(sm.smoothed_state[1, -1]) if sm.smoothed_state.shape[0] > 1 else 0.0

        return FittedModel(
            name="LocalLinearTrend",
            log_likelihood=float(result.llf),
            sigma_obs=float(sigma_obs),
            sigma_level=float(sigma_level),
            sigma_slope=float(sigma_slope),
            last_level=last_level,
            last_slope=last_slope,
            last_log_revenue=self.log_rev[-1],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_param(result, name: str, fallback_idx: int) -> float:
        """
        Extract a scalar parameter from a fitted SSM result by name.
        `result.params` is a plain numpy ndarray in statsmodels;
        `result.param_names` is the companion list of names.
        Falls back to positional index if name is not found.
        """
        try:
            pnames = list(result.param_names)
        except AttributeError:
            try:
                pnames = list(result.model.param_names)
            except AttributeError:
                pnames = []

        if name in pnames:
            return float(result.params[pnames.index(name)])
        if len(result.params) > fallback_idx:
            return float(result.params[fallback_idx])
        return 1e-8

    def _fit_ssm(self, mod: UnobservedComponents, name: str):
        """
        Fit a statsmodels state-space model trying three optimisers.
        Returns the result with the highest log-likelihood.
        Raises RuntimeError if all attempts fail.
        """
        best_result, best_llf = None, -np.inf

        for cfg in [
            {"disp": False},
            {"method": "nm",     "disp": False},
            {"method": "powell", "disp": False},
        ]:
            try:
                r = mod.fit(**cfg, maxiter=500)
                if r.llf > best_llf:
                    best_llf, best_result = r.llf, r
            except Exception as exc:
                logger.debug(f"[{name}] optimiser attempt failed: {exc}")

        if best_result is None:
            raise RuntimeError(
                f"Kalman Filter for {name} did not converge. "
                "Inspect your revenue time series for structural breaks or outliers."
            )

        if not best_result.mle_retvals.get("converged", True):
            logger.warning(
                f"[{name}] MLE gradient norm above tolerance. "
                "Using best result (llf={best_llf:.4f})."
            )

        return best_result
