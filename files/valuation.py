"""
Pydantic schemas for SDCF valuation API.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class ModelSelectionResult(BaseModel):
    """Result of LRT-based model selection."""
    selected_model: Literal["AR1", "LocalLevel", "LocalLinearTrend"]
    ar1_log_likelihood: float
    local_level_log_likelihood: float
    local_linear_trend_log_likelihood: float
    lrt_ar1_vs_local_level: float = Field(description="LRT statistic AR(1) vs LocalLevel")
    lrt_local_level_vs_llt: float = Field(description="LRT statistic LocalLevel vs LocalLinearTrend")
    p_value_ar1_vs_ll: float
    p_value_ll_vs_llt: float
    model_parameters: dict

    model_config = ConfigDict(json_schema_extra={"example": {"selected_model": "LocalLevel"}})


class CashFlowParams(BaseModel):
    """Estimated parameters linking Revenue to Free Cash Flow."""
    alpha: float = Field(description="Operating margin (OCF / Revenue regression coefficient)")
    beta: float = Field(description="3-year avg Working Capital / Revenue ratio")
    alpha_r_squared: float = Field(description="R² of the OCF ~ Revenue regression")
    alpha_t_stat: float = Field(description="t-statistic for alpha")
    net_margin: float = Field(description="Effective net cash margin = alpha - beta")


class WACCComponents(BaseModel):
    """Decomposition of WACC calculation."""
    risk_free_rate: float
    equity_risk_premium: float
    beta_levered: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    debt_weight: float
    equity_weight: float
    wacc: float


class DistributionStats(BaseModel):
    """Descriptive statistics for a distribution of simulated values."""
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    histogram_counts: list[int]
    histogram_edges: list[float]


class MisvaluationMetrics(BaseModel):
    """Bottazzi et al. misvaluation metrics."""
    market_price: float
    mean_log_fair_value: float
    std_log_fair_value: float
    z_score: float = Field(description="z = (ln(P_market) - mean(ln(FV))) / std(ln(FV))")
    signal: Literal["UNDERPRICED", "FAIRLY_VALUED", "OVERPRICED"]
    confidence: str = Field(description="Human-readable confidence level")
    probability_undervalued: float = Field(
        description="Fraction of simulations where FairValue > MarketPrice"
    )


class RawFinancials(BaseModel):
    """Summary of raw financial data fetched from yfinance."""
    ticker: str
    years_of_data: int
    latest_revenue: float
    latest_ocf: float
    latest_working_capital: float
    total_debt: float
    cash_and_equivalents: float
    minority_interest: float
    shares_outstanding: float
    market_cap: float
    current_price: float
    currency: str


# ---------------------------------------------------------------------------
# Main response schema
# ---------------------------------------------------------------------------

class SDCFValuationResponse(BaseModel):
    """Full SDCF valuation response as per Bottazzi et al."""
    ticker: str
    status: Literal["success", "partial", "error"]
    message: Optional[str] = None

    # Raw data summary
    financials: RawFinancials

    # Model selection
    model_selection: ModelSelectionResult

    # CF parameters
    cash_flow_params: CashFlowParams

    # WACC
    wacc_components: WACCComponents

    # Simulation outputs
    n_simulations: int
    horizon_years: int
    fair_value_distribution: DistributionStats
    mean_fair_value_per_share: float
    median_fair_value_per_share: float

    # Misvaluation
    misvaluation: MisvaluationMetrics

    model_config = ConfigDict(json_schema_extra={"example": {"ticker": "AAPL", "status": "success"}})


class ErrorResponse(BaseModel):
    ticker: str
    status: Literal["error"]
    message: str
    detail: Optional[str] = None
