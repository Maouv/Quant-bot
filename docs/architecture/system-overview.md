# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│ Data Layer                                                  │
├─────────────────────────────────────────────────────────────┤
│ Binance Vision OHLCV (26 symbols × 1461 days)              │
│ Binance Vision Metrics (optional: OI, L/S ratios)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ IC Research (ic_research.py)                                │
├─────────────────────────────────────────────────────────────┤
│ • Compute 11-14 signals per symbol                          │
│ • Z-score normalize (cross-sectional)                       │
│ • Calculate Spearman IC with Newey-West t-stats             │
│ • Validate: subsample stability, in/out-of-sample           │
│ Output: ic_results.csv, ic_chart.png                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Composite Signals (composite.py)                            │
├─────────────────────────────────────────────────────────────┤
│ • Fast composite (h=1): volatility, liquidity, reversal_1w  │
│ • Slow composite (h=20): volatility, liquidity, momentum    │
│ • Meta composite: 36% Fast + 64% Slow                       │
│ Output: composite_scores.csv, composite_ic.csv              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Portfolio Simulation (portofolio.py)                        │
├─────────────────────────────────────────────────────────────┤
│ • Vol-scaled position sizing                                │
│ • 15% annualized vol targeting                              │
│ • Risk management (loss limits, drawdown)                   │
│ Output: portfolio_pnl.csv, portfolio_metrics.txt             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Load** — OHLCV + metrics from Binance Vision
2. **Compute signals** — 11-14 signals per symbol per day
3. **Preprocess** — Z-score normalize, winsorize
4. **Calculate IC** — Spearman rank correlation (daily, cross-sectional)
5. **Validate** — Subsample stability, in/out-of-sample degradation
6. **Combine** — Fast/Slow/Meta composites
7. **Size positions** — Vol-scaled, signal-proportional
8. **Apply leverage** — 15% vol targeting + regime filter
9. **Simulate P&L** — Daily returns with risk management

## Key Design Decisions

**Cross-sectional IC (not time-series)**
- Captures factor predictability across assets
- Orthogonal to market timing
- Standard in quantitative research

**Spearman rank (not Pearson)**
- Robust to outliers and fat tails in crypto
- No normality assumption required

**Newey-West t-stats (not standard)**
- IC series are autocorrelated
- Corrects for 20-30% overstatement of significance

**Vol-scaled sizing (not equal-weight)**
- Crypto has extreme vol dispersion
- Reduces concentration risk

**Regime filtering (BTC vol scaler)**
- Crypto vol regime shifts dramatically
- Reduces leverage in high-vol, increases in low-vol

## Module Responsibilities

| Module | Input | Process | Output |
|---|---|---|---|
| ic_research.py | OHLCV, metrics | Signal computation, IC calculation | ic_results.csv |
| composite.py | OHLCV, metrics | Signal combination | composite_scores.csv |
| portofolio.py | composite_scores | Position sizing, P&L simulation | portfolio_pnl.csv |

**Status:** Production
