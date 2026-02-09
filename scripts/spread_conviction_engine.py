#!/usr/bin/env python3
"""
===============================================================================
Spread Conviction Engine — Unified Multi-Strategy Vertical Spread Scoring
===============================================================================

Author:     Financial Toolkit (OpenClaw)
Created:    2026-02-09
Version:    1.0.0
License:    MIT

Description:
    A unified conviction engine that scores four vertical spread strategies:

    ┌────────────┬────────┬──────────────────┬────────────────────────────────┐
    │ Strategy   │ Type   │ Philosophy       │ Ideal Setup                    │
    ├────────────┼────────┼──────────────────┼────────────────────────────────┤
    │ bull_put   │ Credit │ Mean Reversion   │ Bullish trend + oversold dip   │
    │ bear_call  │ Credit │ Mean Reversion   │ Bearish trend + overbought rip │
    │ bull_call  │ Debit  │ Breakout         │ Strong bullish momentum        │
    │ bear_put   │ Debit  │ Breakout         │ Strong bearish momentum        │
    └────────────┴────────┴──────────────────┴────────────────────────────────┘

    This extends the original ``advanced_signals.py`` (Bull Put Spread only)
    to a general-purpose spread selection tool.  All four indicator families
    (Ichimoku, RSI, MACD, Bollinger Bands) are computed identically; only
    the *interpretation and scoring weights* change per strategy.

    Credit spreads prioritise **mean-reversion** setups: buying dips
    (bull_put) or selling rips (bear_call) within a prevailing trend.

    Debit spreads prioritise **breakout** setups: strong directional
    momentum confirmed by expanding volatility (Bollinger Bandwidth).

Academic Notes:
    • Ichimoku  → Trend structure & equilibrium (Hosoda, 1968)
    • RSI       → Momentum & mean-reversion potential (Wilder, 1978)
    • MACD      → Trend momentum & acceleration (Appel, 1979)
    • Bollinger → Volatility regime & price envelopes (Bollinger, 2001)
    • Combining orthogonal signals reduces false-positive rate compared to
      any single-indicator strategy (Pring, 2002; Murphy, 1999).

Dependencies:
    pandas >= 2.0, pandas_ta >= 0.4.0, yfinance >= 1.0

Usage:
    $ python3 spread_conviction_engine.py AAPL
    $ python3 spread_conviction_engine.py SPY --strategy bear_call
    $ python3 spread_conviction_engine.py QQQ --strategy bull_call --period 2y
    $ python3 spread_conviction_engine.py AAPL MSFT --strategy bear_put --json

===============================================================================
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Suppress noisy FutureWarnings and deprecation warnings from yfinance/pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")


# =============================================================================
# Constants & Configuration
# =============================================================================

# Indicator parameters — identical to advanced_signals.py.
# Ichimoku uses extended (~3×) periods to filter short-term noise and align
# the cloud with intermediate-term trend structure suitable for multi-week
# options positions.  With senkou=120 and kijun=60, approximately 180 trading
# days of history are required before the cloud is fully populated.
ICHIMOKU_TENKAN: int = 20     # Conversion Line (Tenkan-sen); standard = 9
ICHIMOKU_KIJUN: int = 60      # Base Line (Kijun-sen); standard = 26
ICHIMOKU_SENKOU: int = 120    # Leading Span B (Senkou Span B); standard = 52
ICHIMOKU_CHIKOU: int = 30     # Lagging Span displacement; standard = 26
RSI_LENGTH: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BBANDS_LENGTH: int = 20
BBANDS_STD: float = 2.0


# =============================================================================
# Strategy Framework
# =============================================================================

class StrategyType(str, Enum):
    """
    Supported vertical spread strategies.

    Properties expose directional and credit/debit classification so that
    scoring functions can branch cleanly without string comparisons.
    """
    BULL_PUT = "bull_put"
    BEAR_CALL = "bear_call"
    BULL_CALL = "bull_call"
    BEAR_PUT = "bear_put"

    @property
    def is_bullish(self) -> bool:
        """True for strategies that profit from upward price movement."""
        return self in (StrategyType.BULL_PUT, StrategyType.BULL_CALL)

    @property
    def is_bearish(self) -> bool:
        """True for strategies that profit from downward price movement."""
        return self in (StrategyType.BEAR_CALL, StrategyType.BEAR_PUT)

    @property
    def is_credit(self) -> bool:
        """True for net-credit strategies (mean-reversion philosophy)."""
        return self in (StrategyType.BULL_PUT, StrategyType.BEAR_CALL)

    @property
    def is_debit(self) -> bool:
        """True for net-debit strategies (breakout philosophy)."""
        return self in (StrategyType.BULL_CALL, StrategyType.BEAR_PUT)

    @property
    def label(self) -> str:
        """Human-friendly strategy label for reports."""
        labels = {
            StrategyType.BULL_PUT:  "Bull Put Spread (Credit)",
            StrategyType.BEAR_CALL: "Bear Call Spread (Credit)",
            StrategyType.BULL_CALL: "Bull Call Spread (Debit)",
            StrategyType.BEAR_PUT:  "Bear Put Spread (Debit)",
        }
        return labels[self]

    @property
    def philosophy(self) -> str:
        """Trading philosophy label."""
        return "Mean Reversion" if self.is_credit else "Breakout / Momentum"

    @property
    def ideal_setup(self) -> str:
        """One-line description of the ideal market conditions."""
        setups = {
            StrategyType.BULL_PUT:  "Bullish trend + oversold pullback → bounce expected",
            StrategyType.BEAR_CALL: "Bearish trend + overbought rally → rejection expected",
            StrategyType.BULL_CALL: "Strong bullish momentum + expanding volatility → breakout",
            StrategyType.BEAR_PUT:  "Strong bearish momentum + expanding volatility → breakdown",
        }
        return setups[self]


@dataclass(frozen=True)
class StrategyWeights:
    """
    Per-strategy component weights (must sum to 100).

    Credit spreads give more weight to RSI (mean-reversion entry timing)
    and Ichimoku (trend structure to revert within).

    Debit spreads give more weight to MACD (momentum confirmation) and
    maintain Bollinger weight (bandwidth expansion validates breakout).
    """
    ichimoku: int
    rsi: int
    macd: int
    bollinger: int

    def __post_init__(self) -> None:
        total = self.ichimoku + self.rsi + self.macd + self.bollinger
        assert total == 100, f"Weights must sum to 100, got {total}"


STRATEGY_WEIGHTS: dict[StrategyType, StrategyWeights] = {
    # Credit: Trend structure (30) + entry timing (25) + momentum (20) + vol (25)
    StrategyType.BULL_PUT:  StrategyWeights(ichimoku=30, rsi=25, macd=20, bollinger=25),
    StrategyType.BEAR_CALL: StrategyWeights(ichimoku=30, rsi=25, macd=20, bollinger=25),
    # Debit: Trend confirm (25) + direction (15) + momentum (35) + vol (25)
    StrategyType.BULL_CALL: StrategyWeights(ichimoku=25, rsi=15, macd=35, bollinger=25),
    StrategyType.BEAR_PUT:  StrategyWeights(ichimoku=25, rsi=15, macd=35, bollinger=25),
}


# =============================================================================
# Enumerations — Readable Signal Labels
# =============================================================================

class TrendBias(str, Enum):
    """Qualitative trend classification (objective, strategy-independent)."""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class ConvictionTier(str, Enum):
    """
    Maps the raw 0–100 score to an actionable tier.

    The tiers encode a patience framework:
      - WAIT:    Conditions are poor for this strategy. Do nothing.
      - WATCH:   Getting interesting. Add to watchlist.
      - PREPARE: Conditions are favourable. Size the trade.
      - EXECUTE: High conviction. Enter the spread.
    """
    WAIT = "WAIT"         # 0–39
    WATCH = "WATCH"       # 40–59
    PREPARE = "PREPARE"   # 60–79
    EXECUTE = "EXECUTE"   # 80–100

    @classmethod
    def from_score(cls, score: float) -> "ConvictionTier":
        """Classify a numeric conviction score into an action tier."""
        if score >= 80:
            return cls.EXECUTE
        elif score >= 60:
            return cls.PREPARE
        elif score >= 40:
            return cls.WATCH
        else:
            return cls.WAIT


# =============================================================================
# Data Classes — Structured Signal Output
# =============================================================================

@dataclass
class IchimokuSignal:
    """
    Ichimoku Kinko Hyo signal decomposition.

    Attributes:
        price_vs_cloud:   'ABOVE', 'BELOW', or 'INSIDE'
        tk_cross:         'BULLISH' or 'BEARISH' (Tenkan vs Kijun)
        cloud_color:      'GREEN' if Senkou A > Senkou B, else 'RED'
        cloud_thickness:  Absolute distance between Senkou A and B
        tenkan:           Current Tenkan-sen value
        kijun:            Current Kijun-sen value
        senkou_a:         Current Senkou Span A value
        senkou_b:         Current Senkou Span B value
        component_score:  Sub-score contribution (0 to weight)
    """
    price_vs_cloud: str
    tk_cross: str
    cloud_color: str
    cloud_thickness: float
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float
    component_score: float = 0.0


@dataclass
class RSISignal:
    """
    Relative Strength Index signal.

    Attributes:
        value:            Current RSI reading
        zone:             Human-readable zone label (strategy-specific)
        component_score:  Sub-score contribution (0 to weight)
    """
    value: float
    zone: str
    component_score: float = 0.0


@dataclass
class MACDSignal:
    """
    Moving Average Convergence Divergence signal.

    Attributes:
        macd_value:        MACD line value
        signal_value:      Signal line value
        histogram:         Current histogram bar
        hist_direction:    'RISING', 'FALLING', or 'FLAT'
        crossover:         'BULLISH_CROSS', 'BEARISH_CROSS', or 'NONE'
        macd_above_signal: True if MACD line > Signal line
        component_score:   Sub-score contribution (0 to weight)
    """
    macd_value: float
    signal_value: float
    histogram: float
    hist_direction: str
    crossover: str
    macd_above_signal: bool
    component_score: float = 0.0


@dataclass
class BollingerSignal:
    """
    Bollinger Bands signal.

    Key metrics:
        %B = (Price − Lower) / (Upper − Lower)
            0 → at lower band, 0.5 → at SMA, 1.0 → at upper band
        Bandwidth = (Upper − Lower) / Middle × 100

    Attributes:
        upper:            Upper Bollinger Band
        middle:           Middle Band (SMA)
        lower:            Lower Bollinger Band
        percent_b:        %B value
        bandwidth:        Normalised bandwidth
        component_score:  Sub-score contribution (0 to weight)
    """
    upper: float
    middle: float
    lower: float
    percent_b: float
    bandwidth: float
    component_score: float = 0.0


@dataclass
class ConvictionResult:
    """
    Final output of the Spread Conviction Engine.

    Combines all four indicator signals into one conviction assessment
    for the selected strategy.

    Attributes:
        ticker:           Symbol analysed
        strategy:         Strategy type string
        strategy_label:   Human-friendly strategy name
        price:            Latest closing price
        conviction_score: Aggregate score (0–100)
        tier:             Action tier (WAIT / WATCH / PREPARE / EXECUTE)
        trend_bias:       Overall qualitative trend assessment
        ichimoku:         Detailed Ichimoku signal
        rsi:              Detailed RSI signal
        macd:             Detailed MACD signal
        bollinger:        Detailed Bollinger signal
        rationale:        Human-readable explanation of the score
    """
    ticker: str
    strategy: str
    strategy_label: str
    price: float
    conviction_score: float
    tier: str
    trend_bias: str
    ichimoku: IchimokuSignal
    rsi: RSISignal
    macd: MACDSignal
    bollinger: BollingerSignal
    rationale: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (JSON-safe)."""
        return asdict(self)


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters:
        ticker:   Stock symbol (e.g. 'AAPL', 'SPY')
        period:   Lookback period ('6mo', '1y', '2y', '5y', 'max').
                  Default '2y' ensures sufficient data for extended Ichimoku.
        interval: Candle interval ('1h', '1d', '1wk')

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume

    Raises:
        ValueError: If no data is returned for the given ticker.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         f"Check symbol validity and market hours.")

    # Flatten MultiIndex columns that yfinance sometimes creates
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# =============================================================================
# Indicator Computation
# =============================================================================

def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ichimoku Kinko Hyo indicators and merge into the DataFrame.

    Column names are auto-generated by pandas_ta based on parameter values:
        ITS_20, IKS_60, ISA_20, ISB_60, ICS_60
    """
    ichimoku_df, _ = ta.ichimoku(
        df["High"], df["Low"], df["Close"],
        tenkan=ICHIMOKU_TENKAN,
        kijun=ICHIMOKU_KIJUN,
        senkou=ICHIMOKU_SENKOU,
    )
    return pd.concat([df, ichimoku_df], axis=1)


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI and add as a column."""
    df["RSI"] = ta.rsi(df["Close"], length=RSI_LENGTH)
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD (line, signal, histogram) and merge into the DataFrame.

    Adds columns: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    """
    macd_df = ta.macd(df["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    return pd.concat([df, macd_df], axis=1)


def compute_bbands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Bollinger Bands and merge into the DataFrame.

    Adds columns: BBL_{l}_{s}_{s}, BBM_{l}_{s}_{s}, BBU_{l}_{s}_{s},
                  BBB_{l}_{s}_{s} (bandwidth), BBP_{l}_{s}_{s} (%B)
    """
    bbands_df = ta.bbands(df["Close"], length=BBANDS_LENGTH, std=BBANDS_STD)
    return pd.concat([df, bbands_df], axis=1)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline: compute every indicator in sequence.

    Single entry point for indicator computation.
    """
    df = compute_ichimoku(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bbands(df)
    return df


# =============================================================================
# Signal Scoring Functions — Strategy-Aware
# =============================================================================
# Each function extracts relevant indicator values, computes a normalised
# sub-score (0.0–1.0), and scales it by the strategy-specific component
# weight.  The sub-score represents "how favourable are conditions for the
# SELECTED STRATEGY from this indicator's perspective?"
# =============================================================================

def score_ichimoku(
    df: pd.DataFrame,
    price: float,
    strategy: StrategyType,
    weights: StrategyWeights,
) -> IchimokuSignal:
    """
    Score Ichimoku signal for the given strategy.

    Bullish strategies (bull_put, bull_call):
        Price above cloud, bullish TK cross, green cloud = high score.

    Bearish strategies (bear_call, bear_put):
        Price below cloud, bearish TK cross, red cloud = high score.

    Sub-Signal Weights (internal):
    ┌──────────────────────────────┬────────┐
    │ Price vs Cloud               │ 0.40   │
    │ TK Cross direction           │ 0.25   │
    │ Cloud colour                 │ 0.20   │
    │ Cloud thickness (normalised) │ 0.15   │
    └──────────────────────────────┴────────┘

    Parameters:
        df:       DataFrame with Ichimoku columns
        price:    Current closing price
        strategy: Active strategy
        weights:  Component weights for this strategy

    Returns:
        IchimokuSignal with populated component_score
    """
    latest = df.iloc[-1]

    # Dynamic column names from constants (pandas_ta naming convention)
    tenkan = float(latest[f"ITS_{ICHIMOKU_TENKAN}"])
    kijun = float(latest[f"IKS_{ICHIMOKU_KIJUN}"])
    senkou_a = float(latest[f"ISA_{ICHIMOKU_TENKAN}"])
    senkou_b = float(latest[f"ISB_{ICHIMOKU_KIJUN}"])

    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    cloud_green = senkou_a > senkou_b

    # --- Sub-signal: Price vs Cloud ---
    if price > cloud_top:
        cloud_status = "ABOVE"
    elif price < cloud_bottom:
        cloud_status = "BELOW"
    else:
        cloud_status = "INSIDE"

    if strategy.is_bullish:
        # Bullish: above cloud = good
        price_cloud_score = {"ABOVE": 1.0, "INSIDE": 0.3, "BELOW": 0.0}[cloud_status]
    else:
        # Bearish: below cloud = good
        price_cloud_score = {"BELOW": 1.0, "INSIDE": 0.3, "ABOVE": 0.0}[cloud_status]

    # --- Sub-signal: TK Cross ---
    tk_bullish = tenkan > kijun
    tk_label = "BULLISH" if tk_bullish else "BEARISH"

    if strategy.is_bullish:
        tk_score = 1.0 if tk_bullish else 0.0
    else:
        tk_score = 0.0 if tk_bullish else 1.0

    # --- Sub-signal: Cloud Colour ---
    cloud_color_label = "GREEN" if cloud_green else "RED"

    if strategy.is_bullish:
        cloud_score = 1.0 if cloud_green else 0.0
    else:
        cloud_score = 0.0 if cloud_green else 1.0

    # --- Sub-signal: Cloud Thickness ---
    # Normalise thickness relative to price; 5% of price = max score.
    thickness = abs(senkou_a - senkou_b)
    thickness_pct = (thickness / price) if price > 0 else 0.0
    thickness_score = min(thickness_pct / 0.05, 1.0)

    # Penalise thick cloud that works AGAINST the strategy direction.
    # For bullish strategies: thick red cloud is resistance (bad).
    # For bearish strategies: thick green cloud is support (bad).
    if strategy.is_bullish and not cloud_green:
        thickness_score *= 0.3
    elif strategy.is_bearish and cloud_green:
        thickness_score *= 0.3

    # --- Weighted combination ---
    raw = (
        0.40 * price_cloud_score
        + 0.25 * tk_score
        + 0.20 * cloud_score
        + 0.15 * thickness_score
    )
    component_score = round(raw * weights.ichimoku, 2)

    return IchimokuSignal(
        price_vs_cloud=cloud_status,
        tk_cross=tk_label,
        cloud_color=cloud_color_label,
        cloud_thickness=round(thickness, 4),
        tenkan=round(tenkan, 2),
        kijun=round(kijun, 2),
        senkou_a=round(senkou_a, 2),
        senkou_b=round(senkou_b, 2),
        component_score=component_score,
    )


def score_rsi(
    df: pd.DataFrame,
    strategy: StrategyType,
    weights: StrategyWeights,
) -> RSISignal:
    """
    Score RSI for the given strategy.

    Each strategy has a distinct ideal RSI zone:

    bull_put (Credit — oversold bounce):
    ┌────────────┬───────┬──────────────────────────────────────────────┐
    │ RSI Range  │ Score │ Rationale                                    │
    ├────────────┼───────┼──────────────────────────────────────────────┤
    │ 30–45      │ 1.00  │ Oversold in uptrend = prime bounce setup     │
    │ 45–55      │ 0.80  │ Neutral-bullish. Steady.                     │
    │ 55–65      │ 0.60  │ Bullish momentum. Acceptable.                │
    │ 25–30      │ 0.40  │ Deep oversold. Could bounce or break.        │
    │ 65–75      │ 0.30  │ Overbought caution.                          │
    │ <25        │ 0.10  │ Extreme oversold. Breakdown risk.             │
    │ >75        │ 0.10  │ Extreme overbought. Reversal risk.            │
    └────────────┴───────┴──────────────────────────────────────────────┘

    bear_call (Credit — overbought rejection):
    ┌────────────┬───────┬──────────────────────────────────────────────┐
    │ 55–70      │ 1.00  │ Overbought in downtrend = rejection setup    │
    │ 45–55      │ 0.80  │ Neutral-bearish. Steady.                     │
    │ 35–45      │ 0.60  │ Bearish momentum. Acceptable.                │
    │ 70–75      │ 0.40  │ Deep overbought. Could reject or break up.   │
    │ 25–35      │ 0.30  │ Oversold caution. Bounce risk.               │
    │ >75        │ 0.10  │ Extreme overbought. Blow-off risk.            │
    │ <25        │ 0.10  │ Extreme oversold. Already broken.             │
    └────────────┴───────┴──────────────────────────────────────────────┘

    bull_call (Debit — bullish breakout momentum):
    ┌────────────┬───────┬──────────────────────────────────────────────┐
    │ 55–70      │ 1.00  │ Strong bullish momentum. Ideal.              │
    │ 45–55      │ 0.70  │ Building momentum. Good.                     │
    │ 70–80      │ 0.50  │ Very strong. Approaching extreme.            │
    │ 35–45      │ 0.25  │ Weak momentum. Insufficient.                 │
    │ >80        │ 0.15  │ Parabolic / blow-off top risk.               │
    │ <35        │ 0.05  │ Wrong direction entirely.                     │
    └────────────┴───────┴──────────────────────────────────────────────┘

    bear_put (Debit — bearish breakdown momentum):
    ┌────────────┬───────┬──────────────────────────────────────────────┐
    │ 30–45      │ 1.00  │ Strong bearish momentum. Ideal.              │
    │ 45–55      │ 0.70  │ Building bearish pressure. Good.             │
    │ 20–30      │ 0.50  │ Very strong bearish. Approaching extreme.    │
    │ 55–65      │ 0.25  │ Weak bearish. Insufficient.                  │
    │ <20        │ 0.15  │ Capitulation — potential snap-back risk.     │
    │ >65        │ 0.05  │ Wrong direction entirely.                     │
    └────────────┴───────┴──────────────────────────────────────────────┘

    Parameters:
        df:       DataFrame with 'RSI' column
        strategy: Active strategy
        weights:  Component weights for this strategy

    Returns:
        RSISignal with populated component_score
    """
    rsi_val = float(df.iloc[-1]["RSI"])

    if strategy == StrategyType.BULL_PUT:
        if 30 <= rsi_val <= 45:
            raw, zone = 1.00, "OVERSOLD_BOUNCE (30–45)"
        elif 45 < rsi_val <= 55:
            raw, zone = 0.80, "NEUTRAL_BULLISH (45–55)"
        elif 55 < rsi_val <= 65:
            raw, zone = 0.60, "BULLISH (55–65)"
        elif 25 <= rsi_val < 30:
            raw, zone = 0.40, "DEEP_OVERSOLD (25–30)"
        elif 65 < rsi_val <= 75:
            raw, zone = 0.30, "OVERBOUGHT_CAUTION (65–75)"
        elif rsi_val < 25:
            raw, zone = 0.10, "EXTREME_OVERSOLD (<25)"
        else:  # > 75
            raw, zone = 0.10, "EXTREME_OVERBOUGHT (>75)"

    elif strategy == StrategyType.BEAR_CALL:
        if 55 <= rsi_val <= 70:
            raw, zone = 1.00, "OVERBOUGHT_REJECTION (55–70)"
        elif 45 <= rsi_val < 55:
            raw, zone = 0.80, "NEUTRAL_BEARISH (45–55)"
        elif 35 <= rsi_val < 45:
            raw, zone = 0.60, "BEARISH (35–45)"
        elif 70 < rsi_val <= 75:
            raw, zone = 0.40, "DEEP_OVERBOUGHT (70–75)"
        elif 25 <= rsi_val < 35:
            raw, zone = 0.30, "OVERSOLD_CAUTION (25–35)"
        elif rsi_val > 75:
            raw, zone = 0.10, "EXTREME_OVERBOUGHT (>75)"
        else:  # < 25
            raw, zone = 0.10, "EXTREME_OVERSOLD (<25)"

    elif strategy == StrategyType.BULL_CALL:
        if 55 <= rsi_val <= 70:
            raw, zone = 1.00, "STRONG_BULLISH_MOMENTUM (55–70)"
        elif 45 <= rsi_val < 55:
            raw, zone = 0.70, "BUILDING_MOMENTUM (45–55)"
        elif 70 < rsi_val <= 80:
            raw, zone = 0.50, "VERY_STRONG (70–80)"
        elif 35 <= rsi_val < 45:
            raw, zone = 0.25, "WEAK_MOMENTUM (35–45)"
        elif rsi_val > 80:
            raw, zone = 0.15, "PARABOLIC_RISK (>80)"
        else:  # < 35
            raw, zone = 0.05, "WRONG_DIRECTION (<35)"

    else:  # BEAR_PUT
        if 30 <= rsi_val <= 45:
            raw, zone = 1.00, "STRONG_BEARISH_MOMENTUM (30–45)"
        elif 45 < rsi_val <= 55:
            raw, zone = 0.70, "BUILDING_BEARISH (45–55)"
        elif 20 <= rsi_val < 30:
            raw, zone = 0.50, "VERY_STRONG_BEARISH (20–30)"
        elif 55 < rsi_val <= 65:
            raw, zone = 0.25, "WEAK_BEARISH (55–65)"
        elif rsi_val < 20:
            raw, zone = 0.15, "CAPITULATION_RISK (<20)"
        else:  # > 65
            raw, zone = 0.05, "WRONG_DIRECTION (>65)"

    component_score = round(raw * weights.rsi, 2)

    return RSISignal(
        value=round(rsi_val, 2),
        zone=zone,
        component_score=component_score,
    )


def _detect_crossover(df: pd.DataFrame, macd_col: str, signal_col: str, lookback: int = 3):
    """
    Detect MACD/Signal crossover within the last ``lookback`` bars.

    Returns:
        Tuple of (crossover_label, is_bullish_cross, is_bearish_cross)
    """
    n = min(lookback + 1, len(df))
    for i in range(2, n + 1):
        row_curr = df.iloc[-i + 1]
        row_prev = df.iloc[-i]
        prev_macd = float(row_prev[macd_col])
        prev_sig = float(row_prev[signal_col])
        curr_macd = float(row_curr[macd_col])
        curr_sig = float(row_curr[signal_col])

        if prev_macd <= prev_sig and curr_macd > curr_sig:
            return "BULLISH_CROSS", True, False
        elif prev_macd >= prev_sig and curr_macd < curr_sig:
            return "BEARISH_CROSS", False, True

    return "NONE", False, False


def score_macd(
    df: pd.DataFrame,
    strategy: StrategyType,
    weights: StrategyWeights,
) -> MACDSignal:
    """
    Score MACD for the given strategy.

    Credit strategies (mean reversion) focus on *decelerating* adverse
    momentum — a rising histogram in a dip (bull_put) or a falling
    histogram in a rally (bear_call).

    Debit strategies (breakout) focus on *accelerating* favourable
    momentum — both the MACD line position and histogram strength matter.

    ── Credit Sub-Signal Weights ──────────────────────────────────────
    │ MACD vs Signal (favourable side)   │ 0.40                       │
    │ Histogram direction (decelerating) │ 0.35                       │
    │ Recent favourable crossover        │ 0.25                       │
    ──────────────────────────────────────────────────────────────────

    ── Debit Sub-Signal Weights ───────────────────────────────────────
    │ MACD vs Signal + zero-line         │ 0.30                       │
    │ Histogram strength (dir + sign)    │ 0.45                       │
    │ Recent favourable crossover        │ 0.25                       │
    ──────────────────────────────────────────────────────────────────

    Parameters:
        df:       DataFrame with MACD columns
        strategy: Active strategy
        weights:  Component weights for this strategy

    Returns:
        MACDSignal with populated component_score
    """
    macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    macd_val = float(latest[macd_col])
    signal_val = float(latest[signal_col])
    hist_val = float(latest[hist_col])
    prev_hist = float(prev[hist_col])

    macd_above = macd_val > signal_val

    # --- Histogram direction ---
    hist_diff = hist_val - prev_hist
    if abs(hist_diff) < 0.001:
        hist_direction = "FLAT"
    elif hist_diff > 0:
        hist_direction = "RISING"
    else:
        hist_direction = "FALLING"

    # --- Crossover detection ---
    crossover_label, is_bull_cross, is_bear_cross = _detect_crossover(
        df, macd_col, signal_col, lookback=3
    )

    # --- Strategy-specific scoring ---
    if strategy.is_credit:
        # ── Credit: Mean Reversion ───────────────────────────────────
        # bull_put wants: MACD above signal, rising histogram, bullish cross
        # bear_call wants: MACD below signal, falling histogram, bearish cross
        if strategy.is_bullish:
            # bull_put
            above_score = 1.0 if macd_above else 0.0
            hist_dir_score = (
                1.0 if hist_direction == "RISING"
                else 0.4 if hist_direction == "FLAT"
                else 0.0
            )
            cross_score = (
                1.0 if is_bull_cross
                else 0.0 if is_bear_cross
                else 0.2
            )
        else:
            # bear_call (inverted)
            above_score = 0.0 if macd_above else 1.0
            hist_dir_score = (
                1.0 if hist_direction == "FALLING"
                else 0.4 if hist_direction == "FLAT"
                else 0.0
            )
            cross_score = (
                1.0 if is_bear_cross
                else 0.0 if is_bull_cross
                else 0.2
            )

        raw = (
            0.40 * above_score
            + 0.35 * hist_dir_score
            + 0.25 * cross_score
        )

    else:
        # ── Debit: Breakout / Momentum ───────────────────────────────
        # Needs strong, *accelerating* directional momentum.
        if strategy.is_bullish:
            # bull_call
            # Sub-signal 1: MACD above signal + positive territory
            if macd_above and macd_val > 0:
                position_score = 1.0
            elif macd_above and macd_val <= 0:
                position_score = 0.5
            else:
                position_score = 0.0

            # Sub-signal 2: Histogram positive AND rising
            if hist_val > 0 and hist_direction == "RISING":
                hist_strength_score = 1.0
            elif hist_val > 0 and hist_direction == "FALLING":
                hist_strength_score = 0.35
            elif hist_val <= 0 and hist_direction == "RISING":
                hist_strength_score = 0.25
            else:  # negative and falling
                hist_strength_score = 0.0

            # Sub-signal 3: Crossover
            cross_score = (
                1.0 if is_bull_cross
                else 0.0 if is_bear_cross
                else 0.15
            )
        else:
            # bear_put (mirror of bull_call)
            # Sub-signal 1: MACD below signal + negative territory
            if not macd_above and macd_val < 0:
                position_score = 1.0
            elif not macd_above and macd_val >= 0:
                position_score = 0.5
            else:
                position_score = 0.0

            # Sub-signal 2: Histogram negative AND falling
            if hist_val < 0 and hist_direction == "FALLING":
                hist_strength_score = 1.0
            elif hist_val < 0 and hist_direction == "RISING":
                hist_strength_score = 0.35
            elif hist_val >= 0 and hist_direction == "FALLING":
                hist_strength_score = 0.25
            else:  # positive and rising
                hist_strength_score = 0.0

            # Sub-signal 3: Crossover
            cross_score = (
                1.0 if is_bear_cross
                else 0.0 if is_bull_cross
                else 0.15
            )

        raw = (
            0.30 * position_score
            + 0.45 * hist_strength_score
            + 0.25 * cross_score
        )

    component_score = round(raw * weights.macd, 2)

    return MACDSignal(
        macd_value=round(macd_val, 4),
        signal_value=round(signal_val, 4),
        histogram=round(hist_val, 4),
        hist_direction=hist_direction,
        crossover=crossover_label,
        macd_above_signal=macd_above,
        component_score=component_score,
    )


def score_bollinger(
    df: pd.DataFrame,
    strategy: StrategyType,
    weights: StrategyWeights,
) -> BollingerSignal:
    """
    Score Bollinger Bands for the given strategy.

    Credit strategies want price near a band (support/resistance) with
    *moderate* bandwidth — the mean-reversion sweet spot.

    Debit strategies want price pushing through a band with *expanding*
    bandwidth — confirming a genuine breakout, not a false signal.

    ── bull_put %B (Credit) ───────────────────────────────────────────
    │ 0.20–0.45 → 1.00  Near lower band, holding. Ideal support.      │
    │ 0.45–0.60 → 0.80  Near middle band. Balanced.                    │
    │ 0.10–0.20 → 0.55  Testing lower band. Could bounce or break.    │
    │ 0.60–0.80 → 0.50  Upper half. Less margin of safety.             │
    │ < 0.10    → 0.15  Breaking below bands. Breakdown risk.          │
    │ > 0.80    → 0.25  Near upper band. Overextended.                  │
    ────────────────────────────────────────────────────────────────────

    ── bear_call %B (Credit) ──────────────────────────────────────────
    │ 0.55–0.80 → 1.00  Near upper band, rejecting. Ideal resistance.  │
    │ 0.40–0.55 → 0.80  Near middle band. Balanced.                    │
    │ 0.80–0.90 → 0.55  Testing upper band. Could reject or break.    │
    │ 0.20–0.40 → 0.50  Lower half. Less margin of safety.             │
    │ > 0.90    → 0.15  Breaking above bands. Breakout risk.           │
    │ < 0.20    → 0.25  Near lower band. Move already made.             │
    ────────────────────────────────────────────────────────────────────

    ── bull_call %B (Debit) ───────────────────────────────────────────
    │ > 0.80    → 1.00  Breaking above upper band. Strong breakout.    │
    │ 0.60–0.80 → 0.85  Upper half, pushing higher. Good.              │
    │ 0.45–0.60 → 0.50  Middle zone. Not yet breaking out.             │
    │ 0.20–0.45 → 0.20  Lower half. Wrong direction.                   │
    │ < 0.20    → 0.05  Near lower band. Completely wrong.              │
    ────────────────────────────────────────────────────────────────────

    ── bear_put %B (Debit) ────────────────────────────────────────────
    │ < 0.20    → 1.00  Breaking below lower band. Strong breakdown.   │
    │ 0.20–0.40 → 0.85  Lower half, pushing lower. Good.               │
    │ 0.40–0.55 → 0.50  Middle zone. Not yet breaking down.            │
    │ 0.55–0.80 → 0.20  Upper half. Wrong direction.                   │
    │ > 0.80    → 0.05  Near upper band. Completely wrong.              │
    ────────────────────────────────────────────────────────────────────

    Bandwidth Scoring:
        Credit:  Moderate (3–10) is ideal; extremes are penalised.
        Debit:   Expanding (>7) is ideal; tight squeezes are uncertain.

    Parameters:
        df:       DataFrame with Bollinger Band columns
        strategy: Active strategy
        weights:  Component weights for this strategy

    Returns:
        BollingerSignal with populated component_score
    """
    bb_suffix = f"{BBANDS_LENGTH}_{BBANDS_STD}_{BBANDS_STD}"
    latest = df.iloc[-1]

    upper = float(latest[f"BBU_{bb_suffix}"])
    middle = float(latest[f"BBM_{bb_suffix}"])
    lower = float(latest[f"BBL_{bb_suffix}"])
    percent_b = float(latest[f"BBP_{bb_suffix}"])
    bandwidth = float(latest[f"BBB_{bb_suffix}"])

    # --- Sub-signal 1: %B positioning (strategy-specific) ---

    if strategy == StrategyType.BULL_PUT:
        if 0.20 <= percent_b <= 0.45:
            pctb_score = 1.00
        elif 0.45 < percent_b <= 0.60:
            pctb_score = 0.80
        elif 0.10 <= percent_b < 0.20:
            pctb_score = 0.55
        elif 0.60 < percent_b <= 0.80:
            pctb_score = 0.50
        elif percent_b < 0.10:
            pctb_score = 0.15
        else:  # > 0.80
            pctb_score = 0.25

    elif strategy == StrategyType.BEAR_CALL:
        if 0.55 <= percent_b <= 0.80:
            pctb_score = 1.00
        elif 0.40 <= percent_b < 0.55:
            pctb_score = 0.80
        elif 0.80 < percent_b <= 0.90:
            pctb_score = 0.55
        elif 0.20 <= percent_b < 0.40:
            pctb_score = 0.50
        elif percent_b > 0.90:
            pctb_score = 0.15
        else:  # < 0.20
            pctb_score = 0.25

    elif strategy == StrategyType.BULL_CALL:
        if percent_b > 0.80:
            pctb_score = 1.00
        elif 0.60 <= percent_b <= 0.80:
            pctb_score = 0.85
        elif 0.45 <= percent_b < 0.60:
            pctb_score = 0.50
        elif 0.20 <= percent_b < 0.45:
            pctb_score = 0.20
        else:  # < 0.20
            pctb_score = 0.05

    else:  # BEAR_PUT
        if percent_b < 0.20:
            pctb_score = 1.00
        elif 0.20 <= percent_b <= 0.40:
            pctb_score = 0.85
        elif 0.40 < percent_b <= 0.55:
            pctb_score = 0.50
        elif 0.55 < percent_b <= 0.80:
            pctb_score = 0.20
        else:  # > 0.80
            pctb_score = 0.05

    # --- Sub-signal 2: Bandwidth regime (credit vs debit) ---

    if strategy.is_credit:
        # Credit: moderate bandwidth is ideal (3–10%)
        if 3.0 <= bandwidth <= 10.0:
            bw_score = 1.0
        elif 10.0 < bandwidth <= 15.0:
            bw_score = 0.6   # Expanding — move already happening
        elif 2.0 <= bandwidth < 3.0:
            bw_score = 0.5   # Squeeze — direction uncertain
        elif bandwidth > 15.0:
            bw_score = 0.3   # Extreme expansion
        else:  # < 2.0
            bw_score = 0.3   # Very tight squeeze
    else:
        # Debit: expanding bandwidth confirms breakout
        if bandwidth > 10.0:
            bw_score = 1.0   # Strong expansion — breakout confirmed
        elif 7.0 <= bandwidth <= 10.0:
            bw_score = 0.80  # Starting to expand
        elif 3.0 <= bandwidth < 7.0:
            bw_score = 0.50  # Moderate — could go either way
        elif 2.0 <= bandwidth < 3.0:
            bw_score = 0.30  # Squeeze — breakout direction unknown
        else:  # < 2.0
            bw_score = 0.20  # Extremely tight — no momentum confirmation

    # --- Weighted combination ---
    # Credit: %B matters more (positioning for mean reversion)
    # Debit:  bandwidth matters more (confirming breakout strength)
    if strategy.is_credit:
        raw = 0.65 * pctb_score + 0.35 * bw_score
    else:
        raw = 0.55 * pctb_score + 0.45 * bw_score

    component_score = round(raw * weights.bollinger, 2)

    return BollingerSignal(
        upper=round(upper, 2),
        middle=round(middle, 2),
        lower=round(lower, 2),
        percent_b=round(percent_b, 4),
        bandwidth=round(bandwidth, 4),
        component_score=component_score,
    )


# =============================================================================
# Trend Classification (Objective — Strategy-Independent)
# =============================================================================

def classify_trend(ichimoku: IchimokuSignal, macd: MACDSignal) -> TrendBias:
    """
    Synthesise Ichimoku and MACD into an overall trend classification.

    This is an *objective* market read — independent of the chosen strategy.
    The conviction score captures whether the trend is *favourable* for a
    given strategy; this function simply labels what the trend IS.

    Logic:
        STRONG_BULL = Above cloud + Bullish TK + MACD above signal + Rising hist
        BULL        = Above cloud + at least one MACD condition
        NEUTRAL     = Mixed signals or inside cloud
        BEAR        = Below cloud + one bearish MACD signal
        STRONG_BEAR = Below cloud + Bearish TK + MACD below + Falling hist
    """
    bull_points = 0

    if ichimoku.price_vs_cloud == "ABOVE":
        bull_points += 2
    elif ichimoku.price_vs_cloud == "INSIDE":
        bull_points += 1

    if ichimoku.tk_cross == "BULLISH":
        bull_points += 1

    if macd.macd_above_signal:
        bull_points += 1

    if macd.hist_direction == "RISING":
        bull_points += 1

    if bull_points >= 5:
        return TrendBias.STRONG_BULL
    elif bull_points >= 3:
        return TrendBias.BULL
    elif bull_points >= 2:
        return TrendBias.NEUTRAL
    elif bull_points >= 1:
        return TrendBias.BEAR
    else:
        return TrendBias.STRONG_BEAR


# =============================================================================
# Rationale Builder
# =============================================================================

def build_rationale(
    strategy: StrategyType,
    weights: StrategyWeights,
    ichimoku: IchimokuSignal,
    rsi: RSISignal,
    macd: MACDSignal,
    bollinger: BollingerSignal,
    trend: TrendBias,
    score: float,
    tier: ConvictionTier,
) -> list[str]:
    """
    Generate a human-readable rationale explaining the conviction score.

    Each line corresponds to a key observation.  This makes the score
    transparent — the trader can always see *why* the engine scored as
    it did, enabling informed manual override when warranted.
    """
    lines: list[str] = []

    # Strategy header
    lines.append(f"Strategy: {strategy.label}  ({strategy.philosophy})")
    lines.append(f"Ideal Setup: {strategy.ideal_setup}")
    lines.append("")
    lines.append(f"Market Trend: {trend.value} | Score: {score:.1f}/100 → {tier.value}")

    # Trend alignment check
    if strategy.is_bullish and trend.value in ("STRONG_BULL", "BULL"):
        lines.append("✅ Trend aligns with bullish strategy")
    elif strategy.is_bearish and trend.value in ("STRONG_BEAR", "BEAR"):
        lines.append("✅ Trend aligns with bearish strategy")
    elif trend.value == "NEUTRAL":
        lines.append("⚠️  Trend is neutral — mixed alignment")
    else:
        lines.append("⚠️  Trend opposes strategy direction — exercise caution")
    lines.append("")

    # Ichimoku
    lines.append(f"[Ichimoku +{ichimoku.component_score:.1f}/{weights.ichimoku}]")
    lines.append(f"  Price is {ichimoku.price_vs_cloud} the cloud")
    lines.append(f"  TK Cross: {ichimoku.tk_cross} "
                 f"(Tenkan {ichimoku.tenkan} vs Kijun {ichimoku.kijun})")
    lines.append(f"  Cloud: {ichimoku.cloud_color}, "
                 f"thickness {ichimoku.cloud_thickness:.2f}")

    # RSI
    lines.append(f"[RSI +{rsi.component_score:.1f}/{weights.rsi}]")
    lines.append(f"  RSI({RSI_LENGTH}) = {rsi.value} → {rsi.zone}")

    # MACD
    lines.append(f"[MACD +{macd.component_score:.1f}/{weights.macd}]")
    direction = "above" if macd.macd_above_signal else "below"
    lines.append(f"  MACD {direction} Signal "
                 f"({macd.macd_value:.4f} vs {macd.signal_value:.4f})")
    lines.append(f"  Histogram: {macd.histogram:.4f} ({macd.hist_direction})")
    if macd.crossover != "NONE":
        lines.append(f"  ⚡ Recent crossover: {macd.crossover}")

    # Bollinger
    lines.append(f"[Bollinger +{bollinger.component_score:.1f}/{weights.bollinger}]")
    lines.append(f"  %B = {bollinger.percent_b:.4f} | "
                 f"Bandwidth = {bollinger.bandwidth:.4f}")
    lines.append(f"  Bands: [{bollinger.lower:.2f} — "
                 f"{bollinger.middle:.2f} — {bollinger.upper:.2f}]")

    return lines


# =============================================================================
# Analysis Engine — The Heart of the System
# =============================================================================

def analyse(
    ticker: str,
    strategy: StrategyType = StrategyType.BULL_PUT,
    period: str = "2y",
    interval: str = "1d",
) -> ConvictionResult:
    """
    Run the full conviction analysis pipeline for a single ticker.

    Pipeline Steps:
        1. Fetch OHLCV data from Yahoo Finance
        2. Compute all four indicator families
        3. Score each indicator (strategy-aware)
        4. Sum component scores → aggregate conviction (0–100)
        5. Classify into action tier
        6. Generate rationale

    Parameters:
        ticker:   Stock symbol
        strategy: Spread strategy to score for (default: bull_put)
        period:   Data lookback period (default '2y')
        interval: Candle interval (default '1d')

    Returns:
        ConvictionResult with all signals and the final score

    Raises:
        ValueError: If data cannot be fetched
        KeyError: If expected indicator columns are missing
    """
    weights = STRATEGY_WEIGHTS[strategy]

    # Step 1: Fetch data
    df = fetch_ohlcv(ticker, period=period, interval=interval)

    # Step 2: Compute indicators
    df = compute_all_indicators(df)

    # Step 3: Get current price
    price = round(float(df.iloc[-1]["Close"]), 2)

    # Step 4: Score each component (strategy-aware)
    ichimoku_sig = score_ichimoku(df, price, strategy, weights)
    rsi_sig = score_rsi(df, strategy, weights)
    macd_sig = score_macd(df, strategy, weights)
    bollinger_sig = score_bollinger(df, strategy, weights)

    # Step 5: Aggregate
    conviction = round(
        ichimoku_sig.component_score
        + rsi_sig.component_score
        + macd_sig.component_score
        + bollinger_sig.component_score,
        2,
    )
    conviction = max(0.0, min(100.0, conviction))  # Clamp to [0, 100]

    tier = ConvictionTier.from_score(conviction)
    trend = classify_trend(ichimoku_sig, macd_sig)

    # Step 6: Rationale
    rationale = build_rationale(
        strategy, weights,
        ichimoku_sig, rsi_sig, macd_sig, bollinger_sig,
        trend, conviction, tier,
    )

    return ConvictionResult(
        ticker=ticker.upper(),
        strategy=strategy.value,
        strategy_label=strategy.label,
        price=price,
        conviction_score=conviction,
        tier=tier.value,
        trend_bias=trend.value,
        ichimoku=ichimoku_sig,
        rsi=rsi_sig,
        macd=macd_sig,
        bollinger=bollinger_sig,
        rationale=rationale,
    )


# =============================================================================
# CLI Interface
# =============================================================================

def print_report(result: ConvictionResult) -> None:
    """Pretty-print a conviction report to stdout."""

    tier_emoji = {
        "WAIT": "🔴",
        "WATCH": "🟡",
        "PREPARE": "🟠",
        "EXECUTE": "🟢",
    }
    emoji = tier_emoji.get(result.tier, "⚪")

    print()
    print("=" * 70)
    print(f"  CONVICTION REPORT: {result.ticker}")
    print(f"  Strategy: {result.strategy_label}")
    print("=" * 70)
    print(f"  Price:       ${result.price}")
    print(f"  Trend:       {result.trend_bias}")
    print(f"  Conviction:  {result.conviction_score:.1f} / 100")
    print(f"  Action Tier: {emoji} {result.tier}")
    print("-" * 70)
    for line in result.rationale:
        print(f"  {line}")
    print("=" * 70)
    print()


def main() -> None:
    """
    CLI entry point.

    Examples:
        python3 spread_conviction_engine.py AAPL
        python3 spread_conviction_engine.py SPY --strategy bear_call
        python3 spread_conviction_engine.py QQQ AAPL --strategy bull_call --json
    """
    parser = argparse.ArgumentParser(
        description=(
            "Spread Conviction Engine — "
            "Multi-strategy vertical spread scoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Strategies:\n"
            "  bull_put   Credit spread. Mean reversion: bullish dip.\n"
            "  bear_call  Credit spread. Mean reversion: bearish rally.\n"
            "  bull_call  Debit spread.  Breakout: bullish momentum.\n"
            "  bear_put   Debit spread.  Breakout: bearish momentum.\n"
            "\n"
            "Conviction Tiers:\n"
            "  🔴 WAIT    (0–39)   → Conditions unfavourable. Stay patient.\n"
            "  🟡 WATCH   (40–59)  → Getting interesting. Monitor closely.\n"
            "  🟠 PREPARE (60–79)  → Favourable. Size your trade.\n"
            "  🟢 EXECUTE (80–100) → High conviction. Enter the spread.\n"
        ),
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more stock ticker symbols (e.g., AAPL SPY QQQ)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="bull_put",
        choices=["bull_put", "bear_call", "bull_call", "bear_put"],
        help="Spread strategy to score (default: bull_put)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval: 1h, 1d, 1wk (default: 1d)",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Data lookback: 6mo, 1y, 2y, 5y (default: 2y)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for piping to other tools)",
    )

    args = parser.parse_args()
    strategy = StrategyType(args.strategy)

    results = []
    for ticker in args.tickers:
        try:
            result = analyse(
                ticker,
                strategy=strategy,
                period=args.period,
                interval=args.interval,
            )
            results.append(result)
            if not args.json:
                print_report(result)
        except Exception as e:
            error_msg = f"Error analysing {ticker}: {e}"
            if args.json:
                results.append({"ticker": ticker, "error": str(e)})
            else:
                print(f"\n  ❌ {error_msg}\n", file=sys.stderr)

    if args.json:
        output = []
        for r in results:
            if isinstance(r, ConvictionResult):
                output.append(r.to_dict())
            else:
                output.append(r)
        print(json.dumps(output, indent=2, default=str))


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
