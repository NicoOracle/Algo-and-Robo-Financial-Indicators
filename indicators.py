"""
indicators.py
=============
Shared indicator functions used across all notebooks and Recommendation_Bot.

Each function uses the list + integer index interface:
    prices_series : list[float]   full price history
    date_idx      : int           current bar index (0-based)

Sources
-------
  indicator_1_pe            — Permutation_Entropy.ipynb  cell c07
  indicator_2_macd          — MACD.ipynb                 cell 2
  indicator_5_fibonacci     — Fibonacci_Retracement.ipynb cell 2
  indicator_4_ma_crossover  — Permutation_Entropy.ipynb  cell c10
  indicator_3_rsi           — Permutation_Entropy.ipynb  cell c09
"""

import math
import numpy as np


# =============================================================================
#  Indicator 1 — Permutation Entropy (Chaos / Order Filter)
# =============================================================================

def _permutation_entropy_score(prices, m=3, normalize=True):
    """Compute Permutation Entropy for a price list.

    Ranks the last m prices and builds ordinal fingerprints.
    Shannon entropy over fingerprint frequencies = market disorder.

    Returns float in [0, 1]:  0 = fully ordered,  1 = fully chaotic.
    """
    n = len(prices)
    if n < m:
        return 1.0

    pattern_counts = {}
    for i in range(n - m + 1):
        pattern = tuple(int(x) for x in np.argsort(prices[i:i + m]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    total   = sum(pattern_counts.values())
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        entropy -= p * math.log(p)

    if normalize:
        max_h   = math.log(math.factorial(m))
        entropy = entropy / max_h if max_h > 0 else 0.0

    return entropy


def indicator_1_pe(prices_series, date_idx, window=30, m=3, threshold=0.75):
    """Permutation Entropy regime filter.

    Returns 1 (orderly: PE < threshold) — proceed with directional signals.
    Returns 0 (chaotic: PE >= threshold) — step aside.
    threshold=0.75 allows trading during moderately trending conditions.
    """
    if date_idx < window + m - 1:
        return 0
    recent = prices_series[date_idx - window + 1:date_idx + 1]
    pe     = _permutation_entropy_score(recent, m=m)
    return 1 if pe < threshold else 0


# =============================================================================
#  Indicator 2 — MACD
# =============================================================================

def _ema(arr, period):
    """Exponential Moving Average helper."""
    k   = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def indicator_2_macd(prices_series, date_idx, fast=12, slow=26, signal_period=9):
    """MACD (Moving Average Convergence Divergence).

    Returns  1 if MACD line is above signal line (bullish momentum),
            -1 if below (bearish momentum),
             0 if insufficient data.
    """
    if date_idx < slow + signal_period:
        return 0
    prices      = np.array(prices_series[:date_idx + 1], dtype=float)
    macd_line   = _ema(prices, fast) - _ema(prices, slow)
    signal_line = _ema(macd_line, signal_period)
    if macd_line[-1] > signal_line[-1]:
        return 1
    elif macd_line[-1] < signal_line[-1]:
        return -1
    return 0


# =============================================================================
#  Indicator 3 — RSI (Momentum Mode)
# =============================================================================

def indicator_3_rsi(prices_series, date_idx, period=14, threshold=50):
    """RSI momentum indicator.

    Uses RSI as a momentum signal, not oversold/overbought.
    RSI > 50  =>  gains dominate  =>  bullish momentum  (+1)
    RSI < 50  =>  losses dominate =>  bearish momentum  (-1)

    Returns  1 if RSI > threshold,  -1 if RSI < threshold,  0 if insufficient data.
    """
    if date_idx < period + 1:
        return 0
    prices   = np.array(prices_series[max(0, date_idx - period * 3):date_idx + 1], dtype=float)
    deltas   = np.diff(prices)
    gains    = np.where(deltas > 0, deltas,  0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rsi = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    if rsi > threshold:
        return 1
    elif rsi < threshold:
        return -1
    return 0


# =============================================================================
#  Indicator 4 — Moving Average Crossover
# =============================================================================

def indicator_4_ma_crossover(prices_series, date_idx, short_window=20, long_window=50):
    """Simple Moving Average crossover.

    Returns  1 if short SMA > long SMA (uptrend confirmed),
            -1 if short SMA < long SMA (downtrend confirmed),
             0 if insufficient data.
    A 0.1% buffer suppresses whipsaw around the crossover point.
    """
    if date_idx < long_window:
        return 0
    prices   = np.array(prices_series[:date_idx + 1], dtype=float)
    short_ma = np.mean(prices[-short_window:])
    long_ma  = np.mean(prices[-long_window:])
    if short_ma > long_ma * 1.001:
        return 1
    elif short_ma < long_ma * 0.999:
        return -1
    return 0


# =============================================================================
#  Indicator 5 — Fibonacci Retracement
# =============================================================================

FIBO_LEVELS = [0.236, 0.382, 0.500, 0.618]   # standard retracement ratios
KEY_LEVELS  = {0.382, 0.618}                  # levels that generate a signal


def _fib_retracement_levels(swing_low, swing_high):
    """Return {ratio: price_level} for all Fibonacci levels.
    Levels measured from swing_high downward (0.382 => 38.2% retracement).
    """
    move = swing_high - swing_low
    return {r: swing_high - r * move for r in FIBO_LEVELS}


def indicator_5_fibonacci(prices_series, date_idx, window=60, proximity_pct=0.015):
    """Fibonacci Retracement indicator.

    Finds swing high and swing low over `window` bars.
    Determines trend direction (which extreme formed more recently).
    Signals when price is within proximity_pct of the 38.2% or 61.8% level.

    Returns  1 if near support in uptrend,
            -1 if near resistance in downtrend,
             0 if no key level nearby or insufficient data.
    """
    if date_idx < window:
        return 0

    window_prices = prices_series[date_idx - window + 1:date_idx + 1]
    current_price = prices_series[date_idx]
    swing_high    = max(window_prices)
    swing_low     = min(window_prices)

    if swing_high - swing_low < 1e-9:
        return 0

    high_idx = window_prices.index(swing_high)
    low_idx  = window_prices.index(swing_low)
    uptrend  = high_idx > low_idx

    levels    = _fib_retracement_levels(swing_low, swing_high)
    tolerance = current_price * proximity_pct

    for ratio, level_price in levels.items():
        if ratio not in KEY_LEVELS:
            continue
        if abs(current_price - level_price) <= tolerance:
            return 1 if uptrend else -1

    return 0
