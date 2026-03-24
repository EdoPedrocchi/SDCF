"""
MonteCarloEngine: Runs 10,000 vectorized simulations of revenue paths
and computes the SDCF equity value distribution (Bottazzi et al.).

Architecture
------------
1. Simulate log-revenue paths from selected state-space model (numpy only).
2. Convert to levels via exp().
3. Apply CF formula to get CF paths.
4. Discount CF paths to PV using WACC.
5. Compute EV distribution, then Equity Value = EV - Debt + Cash.
6. Return distribution statistics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from sdcf.models.revenue_modeler import FittedModel
from sdcf.models.cash_flow import CashFlowEstimator

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Output of the Monte Carlo simulation."""
    fair_value_per_share: np.ndarray   # shape (n_sims,)
    ev_distribution: np.ndarray        # shape (n_sims,)
    equity_distribution: np.ndarray    # shape (n_sims,)

    # Descriptive stats
    mean_fv: float
    median_fv: float
    std_fv: float
    skewness: float
    kurtosis: float
    p5: float
    p25: float
    p75: float
    p95: float
    histogram_counts: list[int]
    histogram_edges: list[float]


class MonteCarloEngine:
    """
    Vectorized Monte Carlo for SDCF valuation.

    Parameters
    ----------
    fitted_model : FittedModel
        Output from RevenueModeler.fit_and_select().
    alpha, beta : float
        Cash flow parameters from CashFlowEstimator.
    wacc : float
        Weighted average cost of capital (annualised).
    total_debt : float
        Latest total debt.
    cash : float
        Latest cash & equivalents.
    minority_interest : float
        Latest minority interest (subtracted from equity value).
    shares : float
        Shares outstanding.
    rev_t0 : float
        Most recent annual revenue level.
    n_sims : int
        Number of Monte Carlo paths (default 10,000).
    horizon : int
        Forecast horizon in years (default 10).
    terminal_growth : float
        Perpetuity growth rate for terminal value (default = rfr - 2%).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        fitted_model: FittedModel,
        alpha: float,
        beta: float,
        wacc: float,
        total_debt: float,
        cash: float,
        minority_interest: float,
        shares: float,
        rev_t0: float,
        n_sims: int = 10_000,
        horizon: int = 10,
        terminal_growth: float = 0.025,
        seed: int | None = 42,
    ):
        self.model = fitted_model
        self.alpha = alpha
        self.beta = beta
        self.wacc = wacc
        self.total_debt = total_debt
        self.cash = cash
        self.minority_interest = minority_interest
        self.shares = max(shares, 1.0)
        self.rev_t0 = rev_t0
        self.n_sims = n_sims
        self.horizon = horizon
        self.terminal_growth = terminal_growth
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute all simulation steps and return results."""
        logger.info(
            f"[MC] Running {self.n_sims:,} paths × {self.horizon}yr | "
            f"model={self.model.name}, WACC={self.wacc:.2%}"
        )

        # Step 1: Simulate log-revenue paths
        log_rev_paths = self._simulate_log_revenue()   # (n_sims, horizon)

        # Step 2: Convert to levels
        rev_paths = np.exp(log_rev_paths)

        # Step 3: Cash flow paths
        cf_paths = CashFlowEstimator.apply_cf_formula(
            self.alpha, self.beta, rev_paths, self.rev_t0
        )

        # Step 4: Discount CF to PV
        ev_distribution = self._discount_to_ev(cf_paths, rev_paths)

        # Step 5: Equity value = EV - Debt + Cash - MinorityInterest
        equity_distribution = (
            ev_distribution - self.total_debt + self.cash - self.minority_interest
        )

        # Per-share
        fv_per_share = equity_distribution / self.shares

        # Remove non-positive values (bankruptcy scenarios possible)
        # Keep them in distribution for correct stats but clip at small positive
        fv_per_share = np.where(fv_per_share <= 0, 1e-4, fv_per_share)

        result = self._compute_stats(fv_per_share, ev_distribution, equity_distribution)
        logger.info(
            f"[MC] Done. Mean FV/share={result.mean_fv:.2f}, "
            f"Median={result.median_fv:.2f}, σ={result.std_fv:.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Step 1: Simulate log-revenue paths (fully vectorized)
    # ------------------------------------------------------------------

    def _simulate_log_revenue(self) -> np.ndarray:
        """
        Dispatch to the correct simulation function based on model type.
        Returns array of shape (n_sims, horizon).
        """
        m = self.model
        if m.name == "AR1":
            return self._sim_ar1(m)
        elif m.name == "LocalLevel":
            return self._sim_local_level(m)
        else:  # LocalLinearTrend
            return self._sim_llt(m)

    def _sim_ar1(self, m: FittedModel) -> np.ndarray:
        """
        AR(1) on first differences of log-revenue:
            Δlog(R_t) = mu + phi * Δlog(R_{t-1}) + ε_t

        Vectorised across n_sims × horizon in one shot.
        """
        sigma = max(m.ar1_sigma, 1e-8)
        phi = m.ar1_phi
        mu = m.ar1_mu

        # Draw all innovations at once: shape (n_sims, horizon)
        eps = self.rng.normal(0.0, sigma, size=(self.n_sims, self.horizon))

        # Initial difference: seed with long-run mean
        d0 = mu / (1.0 - phi) if abs(phi) < 1.0 else mu

        # Build differences iteratively (unavoidable due to AR dependency)
        # Uses numpy; shape (n_sims, horizon)
        diff = np.empty((self.n_sims, self.horizon))
        d_prev = np.full(self.n_sims, d0)

        for t in range(self.horizon):
            d_curr = mu + phi * d_prev + eps[:, t]
            diff[:, t] = d_curr
            d_prev = d_curr

        # Cumsum to get log-revenue levels, seeded at last observed
        log_rev = m.last_log_revenue + np.cumsum(diff, axis=1)
        return log_rev

    def _sim_local_level(self, m: FittedModel) -> np.ndarray:
        """
        Local Level model:
            μ_t = μ_{t-1} + η_t,   η_t ~ N(0, σ²_level)
            y_t = μ_t + ε_t,       ε_t ~ N(0, σ²_obs)
        """
        sigma_level = max(m.sigma_level, 1e-8)
        sigma_obs = max(m.sigma_obs, 1e-8)

        # Innovations: (n_sims, horizon)
        eta = self.rng.normal(0.0, sigma_level, size=(self.n_sims, self.horizon))
        eps = self.rng.normal(0.0, sigma_obs, size=(self.n_sims, self.horizon))

        # State: μ_t = μ_{t-1} + η_t (cumulative sum from last filtered level)
        # Shape: (n_sims, horizon)
        level_shocks = np.cumsum(eta, axis=1) + m.last_level

        # Observation: y_t = μ_t + ε_t
        log_rev = level_shocks + eps
        return log_rev

    def _sim_llt(self, m: FittedModel) -> np.ndarray:
        """
        Local Linear Trend:
            μ_t = μ_{t-1} + ν_{t-1} + η_t,  η_t ~ N(0, σ²_level)
            ν_t = ν_{t-1} + ζ_t,             ζ_t ~ N(0, σ²_slope)
            y_t = μ_t + ε_t,                  ε_t ~ N(0, σ²_obs)
        """
        sigma_level = max(m.sigma_level, 1e-8)
        sigma_slope = max(m.sigma_slope, 1e-8)
        sigma_obs = max(m.sigma_obs, 1e-8)

        eta = self.rng.normal(0.0, sigma_level, size=(self.n_sims, self.horizon))
        zeta = self.rng.normal(0.0, sigma_slope, size=(self.n_sims, self.horizon))
        eps = self.rng.normal(0.0, sigma_obs, size=(self.n_sims, self.horizon))

        level = np.empty((self.n_sims, self.horizon))
        slope = np.empty((self.n_sims, self.horizon))

        mu_prev = np.full(self.n_sims, m.last_level)
        nu_prev = np.full(self.n_sims, m.last_slope)

        for t in range(self.horizon):
            nu_t = nu_prev + zeta[:, t]
            mu_t = mu_prev + nu_prev + eta[:, t]
            level[:, t] = mu_t
            slope[:, t] = nu_t
            mu_prev = mu_t
            nu_prev = nu_t

        log_rev = level + eps
        return log_rev

    # ------------------------------------------------------------------
    # Step 4: Discount CF paths to Enterprise Value
    # ------------------------------------------------------------------

    def _discount_to_ev(
        self, cf_paths: np.ndarray, rev_paths: np.ndarray
    ) -> np.ndarray:
        """
        EV = Σ_{t=1}^{T} CF_t / (1+WACC)^t  +  Terminal Value / (1+WACC)^T

        Terminal Value via Gordon Growth Model:
            TV = CF_{T+1} / (WACC - g)   where CF_{T+1} ≈ CF_T × (1 + g)
        """
        wacc = self.wacc
        g = min(self.terminal_growth, wacc - 0.01)  # g must be < WACC

        # Discount factors: shape (horizon,)
        t = np.arange(1, self.horizon + 1, dtype=float)
        discount = 1.0 / (1.0 + wacc) ** t  # (horizon,)

        # PV of explicit CF: (n_sims, horizon) × (horizon,) → (n_sims,)
        pv_cf = np.sum(cf_paths * discount[np.newaxis, :], axis=1)

        # Terminal value at year T
        cf_terminal = cf_paths[:, -1] * (1.0 + g)
        tv = cf_terminal / (wacc - g)
        pv_tv = tv * discount[-1]  # discount at year T

        ev = pv_cf + pv_tv

        # Floor EV at 0 to avoid negative enterprise values from bad paths
        ev = np.maximum(ev, 0.0)
        return ev

    # ------------------------------------------------------------------
    # Step 6: Statistics
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        fv_per_share: np.ndarray,
        ev_distribution: np.ndarray,
        equity_distribution: np.ndarray,
    ) -> SimulationResult:
        """Compute descriptive statistics and histogram."""
        log_fv = np.log(fv_per_share[fv_per_share > 0])

        # Histogram on log scale for better visualization
        counts, edges = np.histogram(fv_per_share, bins=50)

        return SimulationResult(
            fair_value_per_share=fv_per_share,
            ev_distribution=ev_distribution,
            equity_distribution=equity_distribution,
            mean_fv=float(np.mean(fv_per_share)),
            median_fv=float(np.median(fv_per_share)),
            std_fv=float(np.std(fv_per_share)),
            skewness=float(stats.skew(fv_per_share)),
            kurtosis=float(stats.kurtosis(fv_per_share)),
            p5=float(np.percentile(fv_per_share, 5)),
            p25=float(np.percentile(fv_per_share, 25)),
            p75=float(np.percentile(fv_per_share, 75)),
            p95=float(np.percentile(fv_per_share, 95)),
            histogram_counts=counts.tolist(),
            histogram_edges=edges.tolist(),
        )
