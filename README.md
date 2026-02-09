# Options Spread Conviction Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://clawhub.com)

**Multi-regime vertical spread analysis engine using Ichimoku, RSI, MACD, and Bollinger Bands.**

A comprehensive scoring system for options traders that analyzes market conditions and provides actionable conviction scores (0-100) for four vertical spread strategies.

## 🎯 What It Does

Analyzes any stock ticker and scores four vertical spread strategies:

| Strategy | Type | Philosophy | Ideal Setup |
|----------|------|------------|-------------|
| **bull_put** | Credit | Mean Reversion | Bullish trend + oversold dip |
| **bear_call** | Credit | Mean Reversion | Bearish trend + overbought rip |
| **bull_call** | Debit | Breakout | Strong bullish momentum |
| **bear_put** | Debit | Breakout | Strong bearish momentum |

## 📊 Scoring Methodology

### Credit Spreads (bull_put, bear_call)
| Indicator | Weight | Purpose |
|-----------|--------|---------|
| Ichimoku Cloud | 30 pts | Trend structure & equilibrium |
| RSI | 25 pts | Entry timing (mean-reversion) |
| MACD | 20 pts | Momentum confirmation |
| Bollinger Bands | 25 pts | Volatility regime |

### Debit Spreads (bull_call, bear_put)
| Indicator | Weight | Purpose |
|-----------|--------|---------|
| Ichimoku Cloud | 25 pts | Trend confirmation |
| RSI | 15 pts | Directional momentum |
| MACD | 35 pts | Breakout acceleration |
| Bollinger Bands | 25 pts | Bandwidth expansion |

**Total: 100 points**

## 🎚️ Conviction Tiers

| Score | Tier | Action |
|-------|------|--------|
| 80-100 | **EXECUTE** | High conviction — Enter the spread |
| 60-79 | **PREPARE** | Favorable — Size the trade |
| 40-59 | **WATCH** | Interesting — Add to watchlist |
| 0-39 | **WAIT** | Poor conditions — Avoid / No setup |

## 🚀 Installation

### Via ClawHub (Recommended)
```bash
clawhub install options-spread-conviction-engine
conviction-engine AAPL --strategy bull_call
```

### Manual Installation
```bash
git clone https://github.com/YOUR_USERNAME/options-spread-conviction-engine.git
cd options-spread-conviction-engine
bash scripts/setup-venv.sh
./scripts/conviction-engine AAPL
```

## 📖 Usage

### Basic Analysis
```bash
# Analyze AAPL with default strategy (bull_put)
conviction-engine AAPL

# Specific strategy
conviction-engine SPY --strategy bear_call
conviction-engine QQQ --strategy bull_call --period 2y
```

### Multiple Tickers
```bash
conviction-engine AAPL MSFT GOOGL --strategy bull_put
```

### JSON Output (for automation)
```bash
conviction-engine TSLA --strategy bear_call --json
```

### Full Options
```bash
conviction-engine <ticker> [ticker...]
  --strategy {bull_put,bear_call,bull_call,bear_put}
  --period {1y,2y,3y,5y}
  --json
  --verbose
```

## 📈 Example Output

```
======================================================================
  CONVICTION REPORT: AAPL
  Strategy: Bull Call Spread (Debit)
======================================================================
  Price:       $272.19
  Trend:       BULL
  Conviction:  74.0 / 100
  Action Tier: 🟠 PREPARE
----------------------------------------------------------------------
  Strategy: Bull Call Spread (Debit)  (Breakout / Momentum)
  Ideal Setup: Strong bullish momentum + expanding volatility → breakout
  
  Market Trend: BULL | Score: 74.0/100 → PREPARE
  ✅ Trend aligns with bullish strategy
  
  [Ichimoku +18.8/25]
    Price is ABOVE the cloud
    TK Cross: BEARISH (Tenkan 262.17 vs Kijun 266.02)
    Cloud: GREEN, thickness 20.59
  [RSI +15.0/15]
    RSI(14) = 59.1 → STRONG_BULLISH_MOMENTUM (55–70)
  [MACD +17.3/35]
    MACD above Signal (2.4685 vs -0.3866)
    Histogram: 2.8551 (FALLING)
  [Bollinger +22.9/25]
    %B = 0.7886 | Bandwidth = 15.1431
    Bands: [241.05 — 260.79 — 280.54]
======================================================================
```

## 🎓 Academic Foundation

- **Ichimoku Cloud** — Trend structure & equilibrium (Hosoda, 1968)
- **RSI** — Momentum & mean-reversion potential (Wilder, 1978)
- **MACD** — Trend momentum & acceleration (Appel, 1979)
- **Bollinger Bands** — Volatility regime & price envelopes (Bollinger, 2001)

Combining orthogonal signals reduces false-positive rate compared to any single-indicator strategy (Pring, 2002; Murphy, 1999).

## ⚙️ Requirements

- Python 3.10+ (Python 3.14+ supported via pure-Python mode)
- Isolated virtual environment (auto-created on first run)
- Internet connection (fetches data from Yahoo Finance)

### Dependencies
- pandas >= 2.0
- pandas_ta >= 0.4.0 (pure Python mode on 3.14+)
- yfinance >= 1.0
- scipy
- tqdm

**Note:** On Python 3.14+, the engine runs without numba (numba doesn't support 3.14 yet). Performance is slightly reduced but all functionality works correctly.

## 🏗️ Architecture

```
skills/options-spread-conviction-engine/
├── SKILL.md                    # Skill documentation
├── README.md                   # This file
├── _meta.json                  # Skill metadata
├── scripts/
│   ├── conviction-engine       # Main wrapper script
│   ├── setup-venv.sh          # Environment setup
│   ├── spread_conviction_engine.py  # Core engine (~1400 lines)
│   └── numba.py               # Python 3.14+ compatibility shim
└── assets/                     # Additional resources
```

## 🔧 How It Works

1. **Data Fetching** — Downloads OHLCV data from Yahoo Finance
2. **Indicator Computation** — Calculates Ichimoku, RSI, MACD, Bollinger Bands
3. **Strategy-Aware Scoring** — Each indicator scored based on strategy type
4. **Aggregation** — Sums component scores into 0-100 conviction
5. **Tier Classification** — Maps score to actionable tier (WAIT/WATCH/PREPARE/EXECUTE)
6. **Rationale Generation** — Human-readable explanation of the score

## 📝 License

MIT — Part of the Financial Toolkit for OpenClaw

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional strategies (iron condors, calendars)
- More indicators (ATR, VWAP, Volume Profile)
- Backtesting module
- Webhook alerts for high-conviction setups

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not financial advice. Always do your own due diligence before making investment decisions. Past performance does not guarantee future results.

---

**Built with OpenClaw** | **Author:** Leonardo Da Pinchy
