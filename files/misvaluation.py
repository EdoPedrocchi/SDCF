"""
MisvaluationCalculator: Computes the Bottazzi et al. cross-sectional
misvaluation z-score.

Formula (paper §5):
    z = (ln(P_market) - mean(ln(FV_sim))) / std(ln(FV_sim))

Interpretation:
  z < -1.645  →  UNDERPRICED  (5% left tail, 90% CI)
  z > +1.645  →  OVERPRICED   (5% right tail, 90% CI)
  otherwise   →  FAIRLY VALUED
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

Z_THRESHOLD_LOWER = -1.645  # 5th percentile of N(0,1)
Z_THRESHOLD_UPPER = +1.645  # 95th percentile of N(0,1)


@dataclass
class MisvaluationResult:
    market_price: float
    mean_log_fair_value: float
    std_log_fair_value: float
    z_score: float
    signal: str              # "UNDERPRICED" | "FAIRLY_VALUED" | "OVERPRICED"
    confidence: str
    probability_undervalued: float


class MisvaluationCalculator:
    """
    Computes the cross-sectional misvaluation measure.

    Parameters
    ----------
    fair_value_samples : np.ndarray
        Array of simulated fair-value-per-share values from Monte Carlo.
    market_price : float
        Current market price per share.
    """

    def __init__(self, fair_value_samples: np.ndarray, market_price: float):
        self.samples = fair_value_samples[fair_value_samples > 0]
        self.market_price = market_price

        if len(self.samples) == 0:
            raise ValueError("No valid positive fair-value samples to compute z-score.")
        if market_price <= 0:
            raise ValueError(f"Market price must be positive; got {market_price}.")

    def compute(self) -> MisvaluationResult:
        """Compute misvaluation metrics."""
        log_fv = np.log(self.samples)
        mu_log = float(np.mean(log_fv))
        sigma_log = float(np.std(log_fv, ddof=1))
        sigma_log = max(sigma_log, 1e-8)  # avoid division by zero

        log_price = np.log(self.market_price)
        z_score = (log_price - mu_log) / sigma_log

        # Signal
        if z_score < Z_THRESHOLD_LOWER:
            signal = "UNDERPRICED"
            confidence = self._confidence_label(abs(z_score))
        elif z_score > Z_THRESHOLD_UPPER:
            signal = "OVERPRICED"
            confidence = self._confidence_label(abs(z_score))
        else:
            signal = "FAIRLY_VALUED"
            confidence = f"|z|={abs(z_score):.2f} within ±1.645 band"

        # Probability that fair value > market price
        p_undervalued = float(np.mean(self.samples > self.market_price))

        logger.info(
            f"[Misvaluation] z={z_score:.3f} → {signal} "
            f"(P(FV>P)={p_undervalued:.1%})"
        )

        return MisvaluationResult(
            market_price=self.market_price,
            mean_log_fair_value=mu_log,
            std_log_fair_value=sigma_log,
            z_score=float(z_score),
            signal=signal,
            confidence=confidence,
            probability_undervalued=p_undervalued,
        )

    @staticmethod
    def _confidence_label(abs_z: float) -> str:
        if abs_z >= 2.576:
            return f"|z|={abs_z:.2f} — very high confidence (99% CI)"
        elif abs_z >= 1.960:
            return f"|z|={abs_z:.2f} — high confidence (95% CI)"
        else:
            return f"|z|={abs_z:.2f} — moderate confidence (90% CI)"
