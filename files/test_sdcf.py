"""
tests/test_sdcf.py
==================
Comprehensive test suite for the SDCF valuation engine.
Tests each component in isolation and the full pipeline end-to-end.

Run with:  python -m pytest tests/test_sdcf.py -v
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_revenue() -> pd.Series:
    """Realistic growing revenue series (15 years, ~10% CAGR)."""
    rng = np.random.default_rng(0)
    years = list(range(2010, 2025))
    log_rev = np.cumsum(rng.normal(0.10, 0.04, size=15)) + np.log(10e9)
    return pd.Series(np.exp(log_rev), index=years, name="Revenue")


@pytest.fixture
def synthetic_ocf(synthetic_revenue) -> pd.Series:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.02, size=len(synthetic_revenue))
    return (synthetic_revenue * (0.25 + noise)).rename("OCF")


@pytest.fixture
def synthetic_wc(synthetic_revenue) -> pd.Series:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, size=len(synthetic_revenue))
    return (synthetic_revenue * (0.08 + noise)).rename("WorkingCapital")


# ──────────────────────────────────────────────────────────────────────
# 1. DataService (mock path)
# ──────────────────────────────────────────────────────────────────────

class TestDataService:
    def test_mock_data_generation(self):
        from sdcf.services.mock_data import generate_mock_financials
        m = generate_mock_financials("MSFT", years=15)
        assert len(m["revenue"]) == 15
        assert (m["revenue"] > 0).all()
        assert m["total_debt"] > 0

    def test_mock_revenue_growth(self):
        from sdcf.services.mock_data import generate_mock_financials
        m = generate_mock_financials("MSFT", years=15)
        assert m["revenue"].iloc[-1] > m["revenue"].iloc[0]

    def test_mock_all_tickers(self):
        from sdcf.services.mock_data import generate_mock_financials
        for ticker in ["MSFT", "AAPL", "AMZN", "GOOGL", "META", "NVDA", "XYZ"]:
            m = generate_mock_financials(ticker, years=10)
            assert len(m["revenue"]) == 10

    def test_data_service_fallback(self):
        from sdcf.services.data_service import DataService
        ds = DataService("MSFT", lookback_years=10)
        data = ds.fetch()
        assert data.years_available >= 5
        assert data.wacc > 0
        assert data.current_price > 0


# ──────────────────────────────────────────────────────────────────────
# 2. RevenueModeler
# ──────────────────────────────────────────────────────────────────────

class TestRevenueModeler:
    def test_ar1_fits(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        m = RevenueModeler(synthetic_revenue)
        fitted = m._fit_ar1()
        assert fitted.name == "AR1"
        assert np.isfinite(fitted.log_likelihood)
        assert fitted.ar1_sigma > 0

    def test_ar1_phi_stationary(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        assert abs(fitted.ar1_phi) < 1.5

    def test_local_level_fits(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_local_level()
        assert fitted.name == "LocalLevel"
        assert np.isfinite(fitted.log_likelihood)
        assert fitted.sigma_level >= 0

    def test_llt_fits(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_local_linear_trend()
        assert fitted.name == "LocalLinearTrend"
        assert np.isfinite(fitted.log_likelihood)

    def test_lrt_selects_model(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        selected = RevenueModeler(synthetic_revenue).fit_and_select()
        assert selected.name in ("AR1", "LocalLevel", "LocalLinearTrend")

    def test_lrt_p_values_valid(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        s = RevenueModeler(synthetic_revenue).fit_and_select()
        assert 0.0 <= s.lrt_p_value_ar1_vs_ll <= 1.0
        assert 0.0 <= s.lrt_p_value_ll_vs_llt <= 1.0

    def test_all_likelihoods_populated(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        s = RevenueModeler(synthetic_revenue).fit_and_select()
        for k in ("AR1", "LocalLevel", "LocalLinearTrend"):
            assert k in s.all_log_likelihoods

    def test_insufficient_data_raises(self):
        from sdcf.models.revenue_modeler import RevenueModeler
        short = pd.Series([1e9, 2e9, 3e9], index=[2022, 2023, 2024])
        with pytest.raises(ValueError, match="at least"):
            RevenueModeler(short, min_obs=5)

    def test_last_log_revenue_correct(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue).fit_and_select()
        expected = np.log(float(synthetic_revenue.iloc[-1]))
        assert abs(fitted.last_log_revenue - expected) < 1e-8


# ──────────────────────────────────────────────────────────────────────
# 3. CashFlowEstimator
# ──────────────────────────────────────────────────────────────────────

class TestCashFlowEstimator:
    def test_alpha_range(self, synthetic_revenue, synthetic_ocf, synthetic_wc):
        from sdcf.models.cash_flow import CashFlowEstimator
        p = CashFlowEstimator(synthetic_revenue, synthetic_ocf, synthetic_wc).estimate()
        assert 0.01 <= p.alpha <= 0.99

    def test_beta_range(self, synthetic_revenue, synthetic_ocf, synthetic_wc):
        from sdcf.models.cash_flow import CashFlowEstimator
        p = CashFlowEstimator(synthetic_revenue, synthetic_ocf, synthetic_wc).estimate()
        assert -0.30 <= p.beta <= 0.50

    def test_net_margin_identity(self, synthetic_revenue, synthetic_ocf, synthetic_wc):
        from sdcf.models.cash_flow import CashFlowEstimator
        p = CashFlowEstimator(synthetic_revenue, synthetic_ocf, synthetic_wc).estimate()
        assert abs(p.net_margin - (p.alpha - p.beta)) < 1e-10

    def test_high_r_squared(self, synthetic_revenue, synthetic_ocf, synthetic_wc):
        from sdcf.models.cash_flow import CashFlowEstimator
        p = CashFlowEstimator(synthetic_revenue, synthetic_ocf, synthetic_wc).estimate()
        assert p.alpha_r_squared > 0.80

    def test_cf_formula_shape(self, synthetic_revenue, synthetic_ocf, synthetic_wc):
        from sdcf.models.cash_flow import CashFlowEstimator
        p = CashFlowEstimator(synthetic_revenue, synthetic_ocf, synthetic_wc).estimate()
        rev_paths = np.ones((100, 10)) * 50e9
        cf = CashFlowEstimator.apply_cf_formula(p.alpha, p.beta, rev_paths, 50e9)
        assert cf.shape == (100, 10)

    def test_cf_formula_correctness(self):
        """CF_1 = (α-β)*R_1 + β*R_0, CF_2 = (α-β)*R_2 + β*R_1"""
        from sdcf.models.cash_flow import CashFlowEstimator
        alpha, beta, r0 = 0.30, 0.08, 100.0
        paths = np.array([[110.0, 120.0]])
        cf = CashFlowEstimator.apply_cf_formula(alpha, beta, paths, r0)
        assert abs(cf[0, 0] - ((alpha - beta) * 110.0 + beta * 100.0)) < 1e-10
        assert abs(cf[0, 1] - ((alpha - beta) * 120.0 + beta * 110.0)) < 1e-10


# ──────────────────────────────────────────────────────────────────────
# 4. MonteCarloEngine
# ──────────────────────────────────────────────────────────────────────

class TestMonteCarloEngine:
    def _engine(self, fitted, n_sims=300):
        from sdcf.models.monte_carlo import MonteCarloEngine
        return MonteCarloEngine(
            fitted_model=fitted, alpha=0.25, beta=0.08,
            wacc=0.09, total_debt=20e9, cash=10e9,
            minority_interest=0, shares=1e9, rev_t0=50e9,
            n_sims=n_sims, horizon=5, terminal_growth=0.025, seed=42,
        )

    def test_ar1_output_shape(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        r = self._engine(fitted).run()
        assert r.fair_value_per_share.shape == (300,)

    def test_ll_runs(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_local_level()
        r = self._engine(fitted).run()
        assert r.mean_fv > 0

    def test_llt_runs(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_local_linear_trend()
        r = self._engine(fitted).run()
        assert r.mean_fv > 0

    def test_all_fv_positive(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        r = self._engine(fitted, n_sims=500).run()
        assert np.all(r.fair_value_per_share > 0)

    def test_percentile_ordering(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        r = self._engine(fitted, n_sims=500).run()
        assert r.p5 <= r.p25 <= r.median_fv <= r.p75 <= r.p95

    def test_higher_wacc_lower_value(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        from sdcf.models.monte_carlo import MonteCarloEngine
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        def run(wacc):
            return MonteCarloEngine(
                fitted_model=fitted, alpha=0.25, beta=0.08,
                wacc=wacc, total_debt=5e9, cash=5e9,
                minority_interest=0, shares=1e9, rev_t0=50e9,
                n_sims=1000, horizon=5, seed=42,
            ).run().mean_fv
        assert run(0.07) > run(0.13)

    def test_seed_reproducibility(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        r1 = self._engine(fitted).run()
        r2 = self._engine(fitted).run()
        np.testing.assert_array_equal(r1.fair_value_per_share, r2.fair_value_per_share)

    def test_histogram_count(self, synthetic_revenue):
        from sdcf.models.revenue_modeler import RevenueModeler
        fitted = RevenueModeler(synthetic_revenue)._fit_ar1()
        r = self._engine(fitted).run()
        assert sum(r.histogram_counts) == 300
        assert len(r.histogram_edges) == 51


# ──────────────────────────────────────────────────────────────────────
# 5. MisvaluationCalculator
# ──────────────────────────────────────────────────────────────────────

class TestMisvaluationCalculator:
    def _samples(self, mean=100.0, cv=0.15, n=10_000):
        rng = np.random.default_rng(42)
        return np.exp(rng.normal(np.log(mean), cv, size=n))

    def test_underpriced(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        r = MisvaluationCalculator(self._samples(150.0), 90.0).compute()
        assert r.signal == "UNDERPRICED" and r.z_score < -1.645

    def test_overpriced(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        r = MisvaluationCalculator(self._samples(50.0), 150.0).compute()
        assert r.signal == "OVERPRICED" and r.z_score > 1.645

    def test_fairly_valued(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        r = MisvaluationCalculator(self._samples(100.0), 100.0).compute()
        assert r.signal == "FAIRLY_VALUED"

    def test_z_score_formula(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        rng = np.random.default_rng(0)
        log_fv = rng.normal(4.5, 0.2, size=50_000)
        samples = np.exp(log_fv)
        price = np.exp(4.8)
        r = MisvaluationCalculator(samples, price).compute()
        expected_z = (np.log(price) - np.mean(log_fv)) / np.std(log_fv, ddof=1)
        assert abs(r.z_score - expected_z) < 0.01

    def test_probability_range(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        r = MisvaluationCalculator(self._samples(100.0), 100.0).compute()
        assert 0.0 <= r.probability_undervalued <= 1.0

    def test_negative_price_raises(self):
        from sdcf.models.misvaluation import MisvaluationCalculator
        with pytest.raises(ValueError, match="positive"):
            MisvaluationCalculator(self._samples(), -5.0)


# ──────────────────────────────────────────────────────────────────────
# 6. Full pipeline (integration)
# ──────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    @pytest.fixture(scope="class")
    def result(self):
        from sdcf.services.valuation_orchestrator import ValuationOrchestrator
        return ValuationOrchestrator(n_sims=1500, horizon=10).analyze("MSFT")

    @pytest.fixture(scope="class")
    def result_aapl(self):
        from sdcf.services.valuation_orchestrator import ValuationOrchestrator
        return ValuationOrchestrator(n_sims=1500, horizon=10).analyze("AAPL")

    def test_success_status(self, result):
        assert result.status == "success"

    def test_ticker_echoed(self, result):
        assert result.ticker == "MSFT"

    def test_model_selected(self, result):
        assert result.model_selection.selected_model in ("AR1", "LocalLevel", "LocalLinearTrend")

    def test_wacc_in_range(self, result):
        assert 0.04 <= result.wacc_components.wacc <= 0.25

    def test_fair_value_positive(self, result):
        assert result.mean_fair_value_per_share > 0

    def test_distribution_ordering(self, result):
        d = result.fair_value_distribution
        assert d.percentile_5 <= d.percentile_25 <= d.median <= d.percentile_75 <= d.percentile_95

    def test_signal_valid(self, result):
        assert result.misvaluation.signal in ("UNDERPRICED", "FAIRLY_VALUED", "OVERPRICED")

    def test_z_score_finite(self, result):
        assert np.isfinite(result.misvaluation.z_score)

    def test_different_tickers_differ(self, result, result_aapl):
        assert result.mean_fair_value_per_share != result_aapl.mean_fair_value_per_share

    def test_histogram_total(self, result):
        assert sum(result.fair_value_distribution.histogram_counts) == 1500

    def test_json_serializable(self, result):
        d = result.model_dump()
        serialized = json.dumps(d)
        assert len(serialized) > 500
