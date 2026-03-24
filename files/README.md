# SDCF Valuation Engine

**Stochastic Discounted Cash Flow (SDCF) valuation API** implementing the framework from:

> Bottazzi, G., Grazzi, M., Secchi, A., & Tamagni, F. (2019).
> *"Uncertainty in firm valuation and a cross-sectional misvaluation measure."*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI  GET /analyze/{ticker}             │
└──────────────────────────┬──────────────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │  ValuationOrchestrator │
               └───────────┬───────────┘
      ┌──────────┬──────────┼──────────┬────────────────┐
      │          │          │          │                │
 ┌────▼────┐ ┌───▼────┐ ┌──▼──────┐ ┌─▼──────────┐ ┌──▼────────────┐
 │DataServ.│ │Revenue │ │CashFlow │ │Monte Carlo │ │Misvaluation   │
 │yfinance │ │Modeler │ │Estimator│ │Engine      │ │Calculator     │
 │+mock    │ │AR1/LL/ │ │α,β OLS  │ │10k paths   │ │z-score        │
 │fallback │ │LLT+LRT │ │Bottazzi │ │numpy vec.  │ │Bottazzi §5    │
 └─────────┘ └────────┘ └─────────┘ └────────────┘ └───────────────┘
```

---

## Pipeline

### 1 — Data Ingestion (`DataService`)
Fetches 15 years of annual financials (Revenue, OCF, Working Capital, Debt, Cash).
Computes WACC via CAPM + Damodaran ERP (4.72%). Falls back to synthetic data when
yfinance is unavailable (CI, sandboxes, rate limits).

### 2 — Revenue Modelling with LRT Selection (`RevenueModeler`)
Fits three models to log-revenue and picks the best via Likelihood Ratio Test:

```
Model 1 — AR(1):          Δy_t = μ + φ·Δy_{t−1} + ε_t
Model 2 — Local Level:    y_t = μ_t + ε_t;  μ_t = μ_{t−1} + η_t
Model 3 — Local Lin.Trend: adds stochastic slope ν_t
```

LRT hierarchy (Bottazzi §3):
- Step 1: AR(1) vs Local Level  →  χ²(1), α = 5%
- Step 2: Local Level vs LLT   →  χ²(1), α = 5%

### 3 — Cash Flow Reconstruction (`CashFlowEstimator`)
From Bottazzi §4:

```
CF_t = (α − β)·Rev_t + β·Rev_{t−1}

α = OCF/Revenue (OLS regression coefficient)
β = 3-year avg(WorkingCapital / Revenue)
```

### 4 — Monte Carlo (`MonteCarloEngine`)
10,000 vectorised revenue paths → CF paths → EV via Gordon Growth terminal value:

```python
ev = sum(CF_t / (1+WACC)^t)  +  CF_T*(1+g)/(WACC−g) / (1+WACC)^T
equity = ev − debt + cash − minority_interest
fv_per_share = equity / shares
```

### 5 — Misvaluation Signal (`MisvaluationCalculator`)
```
z = (ln(P_market) − E[ln(FV)]) / σ[ln(FV)]

z < −1.645  →  UNDERPRICED  (5% left tail)
z > +1.645  →  OVERPRICED   (5% right tail)
```

---

## API

### `GET /analyze/{ticker}`

```bash
curl http://localhost:8000/analyze/AAPL
curl http://localhost:8000/analyze/MSFT?n_sims=20000&horizon=12&terminal_growth=0.03
```

**Parameters:**
| Name | Default | Range | Description |
|---|---|---|---|
| `n_sims` | 10000 | 1k–100k | Monte Carlo paths |
| `horizon` | 10 | 3–20 | Forecast years |
| `terminal_growth` | 0.025 | 0–0.10 | Gordon growth rate |

**Key response fields:**
```json
{
  "model_selection":   { "selected_model": "AR1", "p_value_ar1_vs_ll": 1.0 },
  "cash_flow_params":  { "alpha": 0.354, "beta": 0.080, "net_margin": 0.274 },
  "wacc_components":   { "wacc": 0.0844, "cost_of_equity": 0.085 },
  "mean_fair_value_per_share": 89.39,
  "fair_value_distribution": { "percentile_5": 79.47, "percentile_95": 100.54 },
  "misvaluation":      { "z_score": 21.56, "signal": "OVERPRICED" }
}
```

---

## Quick Start

```bash
# Local
pip install -r requirements.txt
uvicorn sdcf.main:app --reload --port 8000
open http://localhost:8000/docs

# Tests (55 passing)
PYTHONPATH=. pytest tests/ -v

# Docker
docker-compose up --build
curl http://localhost:8000/analyze/GOOGL | python -m json.tool
```

---

## Project Structure

```
sdcf/
├── main.py                         # FastAPI app
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── services/
│   ├── data_service.py             # yfinance + WACC
│   ├── mock_data.py                # Synthetic fallback profiles
│   └── valuation_orchestrator.py  # Pipeline wiring
├── models/
│   ├── revenue_modeler.py          # AR(1)/LL/LLT + LRT
│   ├── cash_flow.py                # α,β + CF formula
│   ├── monte_carlo.py              # 10k vectorised paths
│   └── misvaluation.py            # z-score
├── schemas/
│   └── valuation.py                # Pydantic I/O schemas
└── tests/
    └── test_sdcf.py               # 55 unit + integration tests
```

---

## Quant Notes

- **Log-transform revenue** — growth rates are multiplicative; log space makes them Gaussian and stabilises Kalman filter estimation.
- **LRT is one-sided** — df=1 per relaxed constraint, matching paper §3 specification.
- **g < WACC enforced** — Gordon model produces infinite TV otherwise; clamped to `WACC − 1%`.
- **Negative equity paths** — preserved in raw distribution; floored at 1e-4 for log-space z-score.
- **Beta clamped to [0.5, 3.0]** — avoids garbage WACC from data errors in yfinance.
