"""
DataService: Fetches and cleans 15 years of financial data from yfinance.

Provides:
  - Annual revenues, OCF, working capital
  - Balance sheet items: debt, cash, minority interest
  - Market data: price, shares, market cap
  - Risk-free rate (5Y Treasury) and basic WACC
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Equity Risk Premium – Damodaran US estimate (update periodically)
ERP_DEFAULT = 0.0472
# Default corporate tax rate
TAX_RATE_DEFAULT = 0.21


@dataclass
class FinancialData:
    """Container for all fetched financial time series."""
    ticker: str
    revenue: pd.Series          # Annual, index = year
    ocf: pd.Series              # Operating Cash Flow, annual
    working_capital: pd.Series  # Current Assets - Current Liabilities, annual
    total_debt: float           # Latest total debt (long + short term)
    cash: float                 # Latest cash & equivalents
    minority_interest: float    # Latest minority interest
    shares_outstanding: float
    market_cap: float
    current_price: float
    currency: str
    # WACC components
    risk_free_rate: float
    beta: float
    wacc: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    debt_weight: float
    equity_weight: float
    # Metadata
    years_available: int = field(init=False)

    def __post_init__(self):
        self.years_available = len(self.revenue)


class DataService:
    """
    Fetches and preprocesses financial data for SDCF valuation.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL")
    lookback_years : int
        Number of historical years to fetch (default 15)
    """

    TREASURY_TICKER = "^FVX"  # 5-Year Treasury Yield Index

    def __init__(self, ticker: str, lookback_years: int = 15):
        self.ticker = ticker.upper().strip()
        self.lookback_years = lookback_years

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch(self) -> FinancialData:
        """Fetch, clean, and return all data needed for valuation."""
        logger.info(f"[DataService] Fetching data for {self.ticker}")

        stock = yf.Ticker(self.ticker)

        # --- Attempt to fetch real financials first ---
        income_stmt = self._get_income_statement(stock)
        cash_flow_stmt = self._get_cashflow(stock)
        balance_sheet = self._get_balance_sheet(stock)

        # --- If yfinance returns empty DataFrames, fall back to mock data ---
        if income_stmt.empty or cash_flow_stmt.empty or balance_sheet.empty:
            logger.warning(
                f"[DataService] yfinance returned empty data for {self.ticker}. "
                "Falling back to synthetic mock data for demonstration purposes."
            )
            return self._fetch_from_mock()

        info = self._safe_info(stock)

        # --- Time series ---
        revenue = self._extract_revenue(income_stmt)
        ocf = self._extract_ocf(cash_flow_stmt)
        working_capital = self._extract_working_capital(balance_sheet)

        # Align all series to common index and interpolate gaps
        revenue, ocf, working_capital = self._align_and_clean(
            revenue, ocf, working_capital
        )

        if len(revenue) < 5:
            raise ValueError(
                f"Insufficient data for {self.ticker}: only {len(revenue)} years "
                "of revenue available (minimum 5 required)."
            )

        # --- Balance sheet snapshot (most recent) ---
        total_debt = self._get_latest_scalar(
            balance_sheet,
            ["Total Debt", "Long Term Debt And Capital Lease Obligation",
             "Long Term Debt", "Total Long Term Debt"],
            default=0.0,
        )
        short_term_debt = self._get_latest_scalar(
            balance_sheet,
            ["Current Debt And Capital Lease Obligation",
             "Current Debt", "Short Long Term Debt"],
            default=0.0,
        )
        total_debt = total_debt + short_term_debt

        cash = self._get_latest_scalar(
            balance_sheet,
            ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments",
             "Cash And Short Term Investments"],
            default=0.0,
        )
        minority_interest = self._get_latest_scalar(
            balance_sheet,
            ["Minority Interest", "Noncontrolling Interest"],
            default=0.0,
        )

        # --- Market data ---
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        shares = float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1)
        market_cap = float(info.get("marketCap") or price * shares)
        currency = str(info.get("currency", "USD"))

        if price == 0:
            hist = stock.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if market_cap == 0:
            market_cap = price * shares

        # --- Risk-free rate (5Y Treasury) ---
        rfr = self._fetch_risk_free_rate()

        # --- WACC ---
        beta = float(info.get("beta") or 1.0)
        beta = max(0.5, min(beta, 3.0))  # cap to sensible range
        wacc_components = self._compute_wacc(
            beta=beta,
            rfr=rfr,
            total_debt=total_debt,
            market_cap=market_cap,
            income_stmt=income_stmt,
        )

        return FinancialData(
            ticker=self.ticker,
            revenue=revenue,
            ocf=ocf,
            working_capital=working_capital,
            total_debt=total_debt,
            cash=cash,
            minority_interest=minority_interest,
            shares_outstanding=shares,
            market_cap=market_cap,
            current_price=price,
            currency=currency,
            risk_free_rate=rfr,
            beta=beta,
            wacc=wacc_components["wacc"],
            cost_of_equity=wacc_components["cost_of_equity"],
            cost_of_debt=wacc_components["cost_of_debt"],
            tax_rate=wacc_components["tax_rate"],
            debt_weight=wacc_components["debt_weight"],
            equity_weight=wacc_components["equity_weight"],
        )

    # ------------------------------------------------------------------
    # Mock data fallback (used when yfinance is blocked / unavailable)
    # ------------------------------------------------------------------

    def _fetch_from_mock(self) -> FinancialData:
        """Generate synthetic but realistic financials via MockDataProvider."""
        from sdcf.services.mock_data import generate_mock_financials

        logger.info(f"[DataService] Generating mock data for {self.ticker}")
        m = generate_mock_financials(self.ticker, years=self.lookback_years)

        rfr = self._fetch_risk_free_rate()
        beta = m["beta"]
        total_capital = m["market_cap"] + m["total_debt"]
        equity_w = m["market_cap"] / total_capital if total_capital > 0 else 0.8
        debt_w = 1.0 - equity_w
        coe = rfr + beta * ERP_DEFAULT
        cod = rfr + 0.015
        wacc = equity_w * coe + debt_w * cod * (1 - TAX_RATE_DEFAULT)
        wacc = float(np.clip(wacc, 0.04, 0.25))

        return FinancialData(
            ticker=self.ticker,
            revenue=m["revenue"],
            ocf=m["ocf"],
            working_capital=m["working_capital"],
            total_debt=m["total_debt"],
            cash=m["cash"],
            minority_interest=m["minority_interest"],
            shares_outstanding=m["shares_outstanding"],
            market_cap=m["market_cap"],
            current_price=m["current_price"],
            currency=m["currency"],
            risk_free_rate=rfr,
            beta=beta,
            wacc=wacc,
            cost_of_equity=coe,
            cost_of_debt=cod,
            tax_rate=TAX_RATE_DEFAULT,
            debt_weight=debt_w,
            equity_weight=equity_w,
        )

    # ------------------------------------------------------------------
    # Private helpers – data extraction
    # ------------------------------------------------------------------

    def _safe_info(self, stock: yf.Ticker) -> dict:
        try:
            return stock.info or {}
        except Exception:
            return {}

    def _get_income_statement(self, stock: yf.Ticker) -> pd.DataFrame:
        for attr in ("financials", "income_stmt", "incomestmt"):
            try:
                df = getattr(stock, attr)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    def _get_cashflow(self, stock: yf.Ticker) -> pd.DataFrame:
        for attr in ("cashflow", "cash_flow", "cashflowstatement"):
            try:
                df = getattr(stock, attr)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    def _get_balance_sheet(self, stock: yf.Ticker) -> pd.DataFrame:
        for attr in ("balance_sheet", "balancesheet"):
            try:
                df = getattr(stock, attr)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    def _extract_revenue(self, income_stmt: pd.DataFrame) -> pd.Series:
        candidates = [
            "Total Revenue", "Revenue", "Net Revenue",
            "Sales", "Total Sales", "Revenues",
        ]
        return self._extract_series(income_stmt, candidates, "Revenue")

    def _extract_ocf(self, cashflow: pd.DataFrame) -> pd.Series:
        candidates = [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Net Cash Provided By Operating Activities",
            "Total Cash From Operating Activities",
        ]
        return self._extract_series(cashflow, candidates, "OCF")

    def _extract_working_capital(self, balance_sheet: pd.DataFrame) -> pd.Series:
        """Working Capital = Current Assets - Current Liabilities."""
        ca = self._extract_series(
            balance_sheet,
            ["Current Assets", "Total Current Assets"],
            "Current Assets",
        )
        cl = self._extract_series(
            balance_sheet,
            ["Current Liabilities", "Total Current Liabilities"],
            "Current Liabilities",
        )
        if ca.empty or cl.empty:
            raise ValueError(f"Cannot compute Working Capital for {self.ticker}: missing CA or CL.")
        idx = ca.index.intersection(cl.index)
        return (ca.loc[idx] - cl.loc[idx]).rename("WorkingCapital")

    def _extract_series(
        self, df: pd.DataFrame, candidates: list[str], label: str
    ) -> pd.Series:
        """Try each candidate row label; return first match as annual time series."""
        if df.empty:
            raise ValueError(f"Empty financial statement while looking for '{label}'.")
        for col in candidates:
            if col in df.index:
                s = df.loc[col].dropna().astype(float)
                # Convert columns (timestamps) to year integers
                s.index = pd.to_datetime(s.index).year
                s = s.sort_index()
                # Keep last `lookback_years` years
                return s.iloc[-self.lookback_years :]
        raise ValueError(
            f"Could not find '{label}' in financials for {self.ticker}. "
            f"Available rows: {list(df.index[:20])}"
        )

    def _get_latest_scalar(
        self,
        balance_sheet: pd.DataFrame,
        candidates: list[str],
        default: float = 0.0,
    ) -> float:
        if balance_sheet.empty:
            return default
        for row in candidates:
            if row in balance_sheet.index:
                vals = balance_sheet.loc[row].dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
        return default

    # ------------------------------------------------------------------
    # Private helpers – alignment & cleaning
    # ------------------------------------------------------------------

    def _align_and_clean(
        self,
        revenue: pd.Series,
        ocf: pd.Series,
        working_capital: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Align series to a common integer-year index, interpolate gaps."""
        combined = pd.DataFrame(
            {"revenue": revenue, "ocf": ocf, "wc": working_capital}
        )
        # Fill interior NaNs by linear interpolation; ffill/bfill at edges
        combined = combined.interpolate(method="linear", limit_direction="both")
        combined = combined.ffill().bfill()
        combined = combined.dropna()

        if combined.empty:
            raise ValueError(f"No aligned financial data available for {self.ticker}.")

        return combined["revenue"], combined["ocf"], combined["wc"]

    # ------------------------------------------------------------------
    # Private helpers – risk-free rate & WACC
    # ------------------------------------------------------------------

    def _fetch_risk_free_rate(self) -> float:
        """Fetch 5-Year Treasury yield via yfinance. Returns annualised rate."""
        try:
            treasury = yf.Ticker(self.TREASURY_TICKER)
            hist = treasury.history(period="5d")
            if not hist.empty:
                raw = float(hist["Close"].iloc[-1])
                # ^FVX is quoted in yield × 10 (e.g. 42.5 → 4.25 %)
                return raw / 100.0
        except Exception as exc:
            logger.warning(f"Could not fetch 5Y Treasury: {exc}. Using 4.25% default.")
        return 0.0425

    def _compute_wacc(
        self,
        beta: float,
        rfr: float,
        total_debt: float,
        market_cap: float,
        income_stmt: pd.DataFrame,
    ) -> dict:
        # --- Cost of equity (CAPM) ---
        erp = ERP_DEFAULT
        cost_of_equity = rfr + beta * erp

        # --- Cost of debt ---
        # Estimate from interest expense / total debt
        interest_expense = 0.0
        if not income_stmt.empty:
            for row in ["Interest Expense", "Net Interest Income", "Interest And Debt Expense"]:
                if row in income_stmt.index:
                    vals = income_stmt.loc[row].dropna()
                    if not vals.empty:
                        interest_expense = abs(float(vals.iloc[0]))
                        break
        if total_debt > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt
            cost_of_debt = max(rfr, min(cost_of_debt, 0.15))  # sanity clamp
        else:
            cost_of_debt = rfr + 0.015  # 150 bps spread default

        # --- Tax rate ---
        tax_rate = TAX_RATE_DEFAULT
        if not income_stmt.empty:
            for t_row, p_row in [
                ("Tax Provision", "Pretax Income"),
                ("Income Tax Expense", "Income Before Tax"),
            ]:
                if t_row in income_stmt.index and p_row in income_stmt.index:
                    taxes = income_stmt.loc[t_row].dropna()
                    pretax = income_stmt.loc[p_row].dropna()
                    idx = taxes.index.intersection(pretax.index)
                    if not idx.empty:
                        t_val = float(taxes.iloc[0])
                        p_val = float(pretax.iloc[0])
                        if p_val > 0:
                            tax_rate = max(0.05, min(t_val / p_val, 0.40))
                    break

        # --- Weights ---
        total_capital = market_cap + total_debt
        equity_weight = market_cap / total_capital if total_capital > 0 else 0.8
        debt_weight = 1.0 - equity_weight

        # --- WACC ---
        wacc = (equity_weight * cost_of_equity) + (
            debt_weight * cost_of_debt * (1 - tax_rate)
        )
        wacc = max(0.04, min(wacc, 0.25))  # clamp to [4%, 25%]

        return {
            "wacc": wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "tax_rate": tax_rate,
            "debt_weight": debt_weight,
            "equity_weight": equity_weight,
        }

    # ------------------------------------------------------------------
    # Mock data fallback
    # ------------------------------------------------------------------

    def _fetch_from_mock(self) -> FinancialData:
        """
        Build a FinancialData object from synthetic mock financials.
        Used when yfinance is unavailable (sandboxes, CI, network blocks).
        Logs a clear warning so users know real data was not fetched.
        """
        from sdcf.services.mock_data import generate_mock_financials

        logger.warning(
            f"[DataService] ⚠️  USING SYNTHETIC DATA for '{self.ticker}'. "
            "Results are illustrative only — not based on real financials."
        )

        mock = generate_mock_financials(self.ticker, years=self.lookback_years)

        rfr = self._fetch_risk_free_rate()
        beta = mock["beta"]
        erp = ERP_DEFAULT
        cost_of_equity = rfr + beta * erp
        cost_of_debt = rfr + 0.015
        tax_rate = TAX_RATE_DEFAULT

        mktcap = mock["market_cap"]
        debt = mock["total_debt"]
        total_capital = mktcap + debt
        equity_weight = mktcap / total_capital if total_capital > 0 else 0.8
        debt_weight = 1.0 - equity_weight
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
        wacc = max(0.04, min(wacc, 0.25))

        return FinancialData(
            ticker=self.ticker,
            revenue=mock["revenue"],
            ocf=mock["ocf"],
            working_capital=mock["working_capital"],
            total_debt=mock["total_debt"],
            cash=mock["cash"],
            minority_interest=mock["minority_interest"],
            shares_outstanding=mock["shares_outstanding"],
            market_cap=mock["market_cap"],
            current_price=mock["current_price"],
            currency=mock["currency"],
            risk_free_rate=rfr,
            beta=beta,
            wacc=wacc,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            tax_rate=tax_rate,
            debt_weight=debt_weight,
            equity_weight=equity_weight,
        )
