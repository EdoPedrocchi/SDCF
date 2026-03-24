"""
MockDataProvider: Generates realistic financial time series for pipeline testing.
Mimics MSFT-like revenue growth, OCF margins, and balance sheet structure.
Used when yfinance is unavailable (CI, sandboxes, offline dev).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Pre-set realistic profiles for common tickers
_PROFILES = {
    # Microsoft FY2024: Rev ~$245B, OCF margin ~38%, WACC ~8.5%
    "MSFT": dict(
        rev_base=245e9, rev_cagr=0.14, rev_noise=0.03,
        ocf_margin=0.38, wc_ratio=0.07,
        total_debt=98e9, cash=75e9, minority_interest=0,
        shares=7.43e9, price=415.0, beta=0.9,
        currency="USD",
    ),
    # Apple FY2024: Rev ~$391B, OCF margin ~28%
    "AAPL": dict(
        rev_base=391e9, rev_cagr=0.09, rev_noise=0.04,
        ocf_margin=0.28, wc_ratio=-0.02,
        total_debt=101e9, cash=65e9, minority_interest=0,
        shares=15.4e9, price=230.0, beta=1.2,
        currency="USD",
    ),
    # Amazon FY2024: Rev ~$575B, thin margins but growing
    "AMZN": dict(
        rev_base=575e9, rev_cagr=0.20, rev_noise=0.05,
        ocf_margin=0.10, wc_ratio=0.04,
        total_debt=68e9, cash=86e9, minority_interest=0,
        shares=10.4e9, price=195.0, beta=1.4,
        currency="USD",
    ),
    # Alphabet FY2024: Rev ~$307B, OCF margin ~27%
    "GOOGL": dict(
        rev_base=307e9, rev_cagr=0.18, rev_noise=0.04,
        ocf_margin=0.27, wc_ratio=0.06,
        total_debt=29e9, cash=108e9, minority_interest=0,
        shares=12.2e9, price=175.0, beta=1.1,
        currency="USD",
    ),
    # Meta FY2024: Rev ~$160B, high margins
    "META": dict(
        rev_base=160e9, rev_cagr=0.22, rev_noise=0.06,
        ocf_margin=0.42, wc_ratio=0.05,
        total_debt=37e9, cash=70e9, minority_interest=0,
        shares=2.55e9, price=620.0, beta=1.35,
        currency="USD",
    ),
    # Nvidia FY2025: Rev ~$130B, explosive growth
    "NVDA": dict(
        rev_base=130e9, rev_cagr=0.55, rev_noise=0.12,
        ocf_margin=0.52, wc_ratio=0.08,
        total_debt=8e9, cash=34e9, minority_interest=0,
        shares=24.4e9, price=120.0, beta=1.7,
        currency="USD",
    ),
    # Generic mid-cap default
    "DEFAULT": dict(
        rev_base=10e9, rev_cagr=0.07, rev_noise=0.04,
        ocf_margin=0.15, wc_ratio=0.10,
        total_debt=8e9, cash=3e9, minority_interest=0,
        shares=500e6, price=45.0, beta=1.0,
        currency="USD",
    ),
}


def generate_mock_financials(
    ticker: str,
    years: int = 15,
    seed: int = 42,
) -> dict:
    """
    Generate a realistic synthetic annual financials dataset.

    Returns a dict matching the structure expected by FinancialData:
        revenue, ocf, working_capital (as pd.Series indexed by year)
        plus scalar balance sheet items.
    """
    rng = np.random.default_rng(seed)
    profile = _PROFILES.get(ticker.upper(), _PROFILES["DEFAULT"])

    p = profile
    current_year = 2024
    years_list = list(range(current_year - years + 1, current_year + 1))

    # Revenue: compound growth + log-normal noise
    rev = np.empty(years)
    rev[0] = p["rev_base"] * (1 + p["rev_cagr"]) ** (-years + 1)
    for t in range(1, years):
        shock = rng.normal(0, p["rev_noise"])
        rev[t] = rev[t - 1] * (1 + p["rev_cagr"] + shock)
    rev = np.maximum(rev, 1e6)

    # OCF: margin with small noise
    ocf_noise = rng.normal(0, 0.02, size=years)
    ocf = rev * (p["ocf_margin"] + ocf_noise)

    # Working capital: wc_ratio × revenue + noise
    wc_noise = rng.normal(0, 0.01, size=years)
    wc = rev * (p["wc_ratio"] + wc_noise)

    return {
        "revenue": pd.Series(rev, index=years_list, name="Revenue"),
        "ocf": pd.Series(ocf, index=years_list, name="OCF"),
        "working_capital": pd.Series(wc, index=years_list, name="WorkingCapital"),
        "total_debt": p["total_debt"],
        "cash": p["cash"],
        "minority_interest": p["minority_interest"],
        "shares_outstanding": p["shares"],
        "market_cap": p["price"] * p["shares"],
        "current_price": p["price"],
        "currency": p["currency"],
        "beta": p["beta"],
    }
