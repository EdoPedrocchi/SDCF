"""
SDCF Valuation API
==================
FastAPI application implementing the Stochastic Discounted Cash Flow
valuation engine based on Bottazzi et al. (2019).

Endpoints
---------
GET /                    Health check
GET /analyze/{ticker}    Full SDCF valuation
GET /docs                Auto-generated OpenAPI documentation
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sdcf.services.valuation_orchestrator import ValuationOrchestrator
from sdcf.schemas.valuation import SDCFValuationResponse, ErrorResponse

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SDCF Valuation API starting up…")
    yield
    logger.info("SDCF Valuation API shutting down.")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SDCF Valuation API",
    description=(
        "Stochastic Discounted Cash Flow valuation engine "
        "implementing Bottazzi et al. (2019) — "
        "'Uncertainty in firm valuation and a cross-sectional misvaluation measure'."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "SDCF Valuation API",
        "status": "ok",
        "paper": "Bottazzi et al. (2019) — Uncertainty in firm valuation",
        "endpoints": ["/analyze/{ticker}", "/docs"],
    }


@app.get(
    "/analyze/{ticker}",
    response_model=SDCFValuationResponse,
    responses={
        200: {"description": "Successful SDCF valuation"},
        422: {"model": ErrorResponse, "description": "Ticker not found or insufficient data"},
        500: {"model": ErrorResponse, "description": "Internal computation error"},
    },
    summary="Run full SDCF valuation for a ticker",
    description=(
        "Fetches up to 15 years of financial data from Yahoo Finance, fits three "
        "revenue models (AR(1), Local Level, Local Linear Trend) and selects the "
        "best via Likelihood Ratio Test, reconstructs Free Cash Flows using the "
        "Bottazzi et al. operating-margin / working-capital framework, runs "
        "10,000 Monte Carlo simulations, and returns the full fair-value "
        "distribution and misvaluation z-score."
    ),
    tags=["Valuation"],
)
async def analyze_ticker(
    ticker: Annotated[
        str,
        Path(
            description="Stock ticker symbol (e.g. AAPL, MSFT, AMZN)",
            min_length=1,
            max_length=10,
            pattern=r"^[A-Za-z0-9.\-]+$",
        ),
    ],
    n_sims: Annotated[
        int,
        Query(description="Number of Monte Carlo simulations", ge=1000, le=100_000),
    ] = 10_000,
    horizon: Annotated[
        int,
        Query(description="Forecast horizon in years", ge=3, le=20),
    ] = 10,
    terminal_growth: Annotated[
        float,
        Query(description="Perpetuity growth rate for terminal value", ge=0.0, le=0.10),
    ] = 0.025,
):
    """
    Full SDCF valuation for the given ticker.

    ### Pipeline
    1. **Data Ingestion** — 15 years of annual financials via yfinance
    2. **Revenue Modelling** — AR(1) / Local Level / Local Linear Trend + LRT
    3. **CF Reconstruction** — Bottazzi α (operating margin) + β (WC ratio)
    4. **Monte Carlo** — 10,000 revenue paths → CF paths → EV distribution
    5. **Misvaluation** — z-score vs current market price

    ### Interpretation
    - **z < −1.645** → Stock appears underpriced (5% significance)
    - **z > +1.645** → Stock appears overpriced (5% significance)
    """
    orchestrator = ValuationOrchestrator(
        n_sims=n_sims,
        horizon=horizon,
        terminal_growth=terminal_growth,
    )

    try:
        result = orchestrator.analyze(ticker)
        return result

    except ValueError as exc:
        logger.warning(f"[{ticker}] Validation error: {exc}")
        raise HTTPException(
            status_code=422,
            detail={
                "ticker": ticker,
                "status": "error",
                "message": str(exc),
            },
        )
    except RuntimeError as exc:
        logger.error(f"[{ticker}] Runtime error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "ticker": ticker,
                "status": "error",
                "message": str(exc),
            },
        )
    except Exception as exc:
        logger.exception(f"[{ticker}] Unexpected error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "ticker": ticker,
                "status": "error",
                "message": f"Unexpected error: {type(exc).__name__}: {exc}",
            },
        )


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "sdcf.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
