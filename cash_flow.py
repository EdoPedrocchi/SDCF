"""
CashFlowEstimator: Implements the paper's link between Revenue and Free Cash Flow.

From Bottazzi et al. (§4):

    CF_t = (α - β) · Rev_t + β · Rev_{t-1}

Where:
  α = operating margin   → OLS: OCF ~ Revenue
  β = 3-year avg of (ΔWorkingCapital / ΔRevenue) — approximation to WC/Rev ratio
      (paper uses ΔWC / ΔRev; we use WC/Rev average as the 3-yr rolling proxy)

This captures:
  - Operating cash generation (α × Rev)
  - Working capital absorption when revenue grows (β × ΔRev = β·Rev_t - β·Rev_{t-1})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


@dataclass
class CashFlowParams:
    alpha: float           # Operating margin coefficient
    beta: float            # Working capital / revenue ratio
    alpha_r_squared: float
    alpha_t_stat: float
    net_margin: float      # alpha - beta


class CashFlowEstimator:
    """
    Estimates α and β from historical financial data.

    Parameters
    ----------
    revenue : pd.Series
        Annual revenue (levels).
    ocf : pd.Series
        Annual operating cash flow.
    working_capital : pd.Series
        Annual working capital (Current Assets - Current Liabilities).
    beta_window : int
        Rolling window for β estimation (default 3, per paper).
    """

    def __init__(
        self,
        revenue: pd.Series,
        ocf: pd.Series,
        working_capital: pd.Series,
        beta_window: int = 3,
    ):
        self.revenue = revenue.astype(float)
        self.ocf = ocf.astype(float)
        self.working_capital = working_capital.astype(float)
        self.beta_window = beta_window

    def estimate(self) -> CashFlowParams:
        """
        Estimate α and β and return CashFlowParams.
        """
        alpha, r2, t_stat = self._estimate_alpha()
        beta = self._estimate_beta()

        net_margin = alpha - beta
        logger.info(
            f"[CashFlow] α={alpha:.4f}, β={beta:.4f}, "
            f"net_margin={net_margin:.4f}, R²={r2:.4f}"
        )

        return CashFlowParams(
            alpha=alpha,
            beta=beta,
            alpha_r_squared=r2,
            alpha_t_stat=t_stat,
            net_margin=net_margin,
        )

    # ------------------------------------------------------------------
    # Alpha estimation: OLS of OCF on Revenue
    # ------------------------------------------------------------------

    def _estimate_alpha(self) -> tuple[float, float, float]:
        """
        OLS regression: OCF_t = α · Rev_t + ε_t (no constant per Bottazzi).

        Returns (alpha, R², t-statistic).
        """
        # Align indices
        idx = self.revenue.index.intersection(self.ocf.index)
        rev = self.revenue.loc[idx].values
        ocf = self.ocf.loc[idx].values

        # Remove pairs where either is NaN / zero
        mask = (rev > 0) & np.isfinite(ocf) & np.isfinite(rev)
        rev = rev[mask]
        ocf = ocf[mask]

        if len(rev) < 3:
            logger.warning("Insufficient data for α OLS. Using OCF/Revenue median.")
            raw_margin = self.ocf / self.revenue
            alpha = float(raw_margin.replace([np.inf, -np.inf], np.nan).median())
            return max(0.01, alpha), 0.0, 0.0

        # OLS without intercept (paper specification)
        result = OLS(ocf, rev).fit()
        alpha = float(result.params[0])
        r2 = float(result.rsquared)
        t_stat = float(result.tvalues[0])

        # Sanity: operating margin must be positive and < 1
        alpha = float(np.clip(alpha, 0.01, 0.99))
        return alpha, r2, t_stat

    # ------------------------------------------------------------------
    # Beta estimation: 3-year avg of WC / Revenue
    # ------------------------------------------------------------------

    def _estimate_beta(self) -> float:
        """
        β = mean over last `beta_window` years of (WorkingCapital / Revenue).

        Bottazzi et al. use the WC-to-revenue ratio as the coefficient
        linking incremental revenue to incremental WC investment.
        """
        idx = self.revenue.index.intersection(self.working_capital.index)
        rev = self.revenue.loc[idx]
        wc = self.working_capital.loc[idx]

        ratio = wc / rev
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

        if ratio.empty:
            logger.warning("Cannot compute β. Defaulting to 0.05.")
            return 0.05

        # Use the most recent `beta_window` observations
        recent = ratio.iloc[-self.beta_window:]
        beta = float(recent.mean())

        # Clamp: β typically 0–0.5; outside that is data noise
        beta = float(np.clip(beta, -0.30, 0.50))
        return beta

    # ------------------------------------------------------------------
    # Apply CF formula to simulated revenue paths
    # ------------------------------------------------------------------

    @staticmethod
    def apply_cf_formula(
        alpha: float,
        beta: float,
        rev_paths: np.ndarray,
        rev_t0: float,
    ) -> np.ndarray:
        """
        Vectorised application of the Bottazzi et al. CF formula.

        CF_t = (α - β) · Rev_t + β · Rev_{t-1}

        Parameters
        ----------
        alpha, beta : floats
        rev_paths : np.ndarray, shape (n_sims, horizon)
            Simulated future revenue levels.
        rev_t0 : float
            Last historical revenue (Rev at t=0).

        Returns
        -------
        cf_paths : np.ndarray, shape (n_sims, horizon)
        """
        n_sims, horizon = rev_paths.shape
        cf_paths = np.empty_like(rev_paths)

        # Rev_{t-1} for t=1 is rev_t0
        rev_prev = np.full(n_sims, rev_t0)

        for t in range(horizon):
            rev_t = rev_paths[:, t]
            cf_paths[:, t] = (alpha - beta) * rev_t + beta * rev_prev
            rev_prev = rev_t

        return cf_paths
