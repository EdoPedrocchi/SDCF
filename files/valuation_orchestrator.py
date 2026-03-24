"""
ValuationOrchestrator: Wires together DataService → RevenueModeler →
CashFlowEstimator → MonteCarloEngine → MisvaluationCalculator and
returns the complete SDCFValuationResponse.
"""
from __future__ import annotations

import logging
import time

from sdcf.services.data_service import DataService
from sdcf.models.revenue_modeler import RevenueModeler
from sdcf.models.cash_flow import CashFlowEstimator
from sdcf.models.monte_carlo import MonteCarloEngine
from sdcf.models.misvaluation import MisvaluationCalculator
from sdcf.schemas.valuation import (
    SDCFValuationResponse,
    ModelSelectionResult,
    CashFlowParams as CashFlowParamsSchema,
    WACCComponents,
    DistributionStats,
    MisvaluationMetrics,
    RawFinancials,
)

logger = logging.getLogger(__name__)


class ValuationOrchestrator:
    """
    End-to-end SDCF valuation pipeline.

    Parameters
    ----------
    n_sims : int
        Monte Carlo simulations (default 10,000).
    horizon : int
        Forecast horizon in years (default 10).
    lookback_years : int
        Historical data window (default 15).
    terminal_growth : float
        Perpetuity growth rate (default 2.5%).
    """

    def __init__(
        self,
        n_sims: int = 10_000,
        horizon: int = 10,
        lookback_years: int = 15,
        terminal_growth: float = 0.025,
    ):
        self.n_sims = n_sims
        self.horizon = horizon
        self.lookback_years = lookback_years
        self.terminal_growth = terminal_growth

    def analyze(self, ticker: str) -> SDCFValuationResponse:
        """Run the complete SDCF analysis and return structured results."""
        t0 = time.time()
        ticker = ticker.upper().strip()
        logger.info(f"=== SDCF Analysis: {ticker} ===")

        # ── 1. Data Ingestion ──────────────────────────────────────────
        ds = DataService(ticker, lookback_years=self.lookback_years)
        data = ds.fetch()
        logger.info(
            f"Fetched {data.years_available} years of data. "
            f"Latest Rev={data.revenue.iloc[-1]/1e9:.2f}B, "
            f"WACC={data.wacc:.2%}"
        )

        # ── 2. Revenue Modelling ───────────────────────────────────────
        modeler = RevenueModeler(data.revenue)
        fitted = modeler.fit_and_select()

        # ── 3. Cash Flow Parameters ────────────────────────────────────
        cf_estimator = CashFlowEstimator(data.revenue, data.ocf, data.working_capital)
        cf_params = cf_estimator.estimate()

        # ── 4. Monte Carlo Simulation ──────────────────────────────────
        mc = MonteCarloEngine(
            fitted_model=fitted,
            alpha=cf_params.alpha,
            beta=cf_params.beta,
            wacc=data.wacc,
            total_debt=data.total_debt,
            cash=data.cash,
            minority_interest=data.minority_interest,
            shares=data.shares_outstanding,
            rev_t0=float(data.revenue.iloc[-1]),
            n_sims=self.n_sims,
            horizon=self.horizon,
            terminal_growth=self.terminal_growth,
        )
        sim_result = mc.run()

        # ── 5. Misvaluation ────────────────────────────────────────────
        mv_calc = MisvaluationCalculator(
            sim_result.fair_value_per_share, data.current_price
        )
        mv_result = mv_calc.compute()

        elapsed = time.time() - t0
        logger.info(f"=== Analysis complete in {elapsed:.1f}s ===")

        # ── 6. Assemble response ───────────────────────────────────────
        return SDCFValuationResponse(
            ticker=ticker,
            status="success",
            message=f"Analysis completed in {elapsed:.1f}s using {fitted.name} model.",

            financials=RawFinancials(
                ticker=ticker,
                years_of_data=data.years_available,
                latest_revenue=float(data.revenue.iloc[-1]),
                latest_ocf=float(data.ocf.iloc[-1]),
                latest_working_capital=float(data.working_capital.iloc[-1]),
                total_debt=data.total_debt,
                cash_and_equivalents=data.cash,
                minority_interest=data.minority_interest,
                shares_outstanding=data.shares_outstanding,
                market_cap=data.market_cap,
                current_price=data.current_price,
                currency=data.currency,
            ),

            model_selection=ModelSelectionResult(
                selected_model=fitted.name,
                ar1_log_likelihood=fitted.all_log_likelihoods.get("AR1", 0.0),
                local_level_log_likelihood=fitted.all_log_likelihoods.get("LocalLevel", 0.0),
                local_linear_trend_log_likelihood=fitted.all_log_likelihoods.get("LocalLinearTrend", 0.0),
                lrt_ar1_vs_local_level=fitted.lrt_statistic_ar1_vs_ll,
                lrt_local_level_vs_llt=fitted.lrt_statistic_ll_vs_llt,
                p_value_ar1_vs_ll=fitted.lrt_p_value_ar1_vs_ll,
                p_value_ll_vs_llt=fitted.lrt_p_value_ll_vs_llt,
                model_parameters=self._extract_model_params(fitted),
            ),

            cash_flow_params=CashFlowParamsSchema(
                alpha=cf_params.alpha,
                beta=cf_params.beta,
                alpha_r_squared=cf_params.alpha_r_squared,
                alpha_t_stat=cf_params.alpha_t_stat,
                net_margin=cf_params.net_margin,
            ),

            wacc_components=WACCComponents(
                risk_free_rate=data.risk_free_rate,
                equity_risk_premium=0.0472,
                beta_levered=data.beta,
                cost_of_equity=data.cost_of_equity,
                cost_of_debt=data.cost_of_debt,
                tax_rate=data.tax_rate,
                debt_weight=data.debt_weight,
                equity_weight=data.equity_weight,
                wacc=data.wacc,
            ),

            n_simulations=self.n_sims,
            horizon_years=self.horizon,

            fair_value_distribution=DistributionStats(
                mean=sim_result.mean_fv,
                median=sim_result.median_fv,
                std=sim_result.std_fv,
                skewness=sim_result.skewness,
                kurtosis=sim_result.kurtosis,
                percentile_5=sim_result.p5,
                percentile_25=sim_result.p25,
                percentile_75=sim_result.p75,
                percentile_95=sim_result.p95,
                histogram_counts=sim_result.histogram_counts,
                histogram_edges=sim_result.histogram_edges,
            ),

            mean_fair_value_per_share=sim_result.mean_fv,
            median_fair_value_per_share=sim_result.median_fv,

            misvaluation=MisvaluationMetrics(
                market_price=mv_result.market_price,
                mean_log_fair_value=mv_result.mean_log_fair_value,
                std_log_fair_value=mv_result.std_log_fair_value,
                z_score=mv_result.z_score,
                signal=mv_result.signal,
                confidence=mv_result.confidence,
                probability_undervalued=mv_result.probability_undervalued,
            ),
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_model_params(fitted) -> dict:
        params = {}
        if fitted.name == "AR1":
            params = {
                "phi": round(fitted.ar1_phi, 6),
                "mu": round(fitted.ar1_mu, 6),
                "sigma": round(fitted.ar1_sigma, 6),
            }
        elif fitted.name == "LocalLevel":
            params = {
                "sigma_level": round(fitted.sigma_level, 6),
                "sigma_obs": round(fitted.sigma_obs, 6),
                "last_filtered_level": round(fitted.last_level, 6),
            }
        else:  # LLT
            params = {
                "sigma_level": round(fitted.sigma_level, 6),
                "sigma_slope": round(fitted.sigma_slope, 6),
                "sigma_obs": round(fitted.sigma_obs, 6),
                "last_filtered_level": round(fitted.last_level, 6),
                "last_filtered_slope": round(fitted.last_slope, 6),
            }
        return params
