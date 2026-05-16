# Quant-Bot: Crypto IC Analysis

Quantitative research system for cryptocurrency futures using cross-sectional Information Coefficient (IC) analysis.

## Quick Start

```bash
pip install pandas numpy scipy matplotlib statsmodels
python3 ic_research.py      # Discover signals
python3 composite.py        # Build composites
python3 portofolio.py       # Simulate positions
```

## System Flow

```
Data (Binance Vision)
  ↓
IC Research → ic_results.csv, ic_chart.png
  ↓
Composite Signals → composite_scores.csv
  ↓
Portfolio Simulation → portfolio_pnl.csv
```

## Key Results

| Signal | IC Mean | t-stat | Status |
|---|---|---|---|
| volatility | +0.068 | 7.07 | ✅ |
| liquidity | -0.042 | -5.66 | ✅ |
| quote_liquidity | -0.039 | -5.42 | ✅ |
| volume_compression | -0.026 | -3.86 | ✅ |
| reversal_1w | +0.023 | 3.01 | ✅ |
| reversal_1d | +0.018 | 2.33 | ✅ |

## Data

- **Exchange:** Binance Futures
- **Range:** 2022-01-01 to 2025-12-31
- **Symbols:** 26 major crypto futures
- **Split:** 70% in-sample (2022-2024), 30% out-of-sample (2024-2025)

## Documentation

- [IC Research](./modules/ic-research.md) — Signal discovery and validation
- [Composite Signals](./modules/composite-signals.md) — Signal weighting and blending
- [Portfolio](./modules/portfolio.md) — Position sizing and risk management
- [Mathematics](./framework/mathematics.md) — IC theory and formulas
- [Data Pipeline](./data/pipeline.md) — Binance Vision integration
- [Development](./development/setup.md) — Setup and testing
- [API Reference](./api/reference.md) — Input/output formats
- [Troubleshooting](./troubleshooting/common-issues.md) — Known issues

## References

- Liu, Tsyvinski & Wu (2022) — Common Risk Factors in Cryptocurrency
- Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers
- Fama & French (2015) — A five-factor asset pricing model

**Status:** Development | **Last Updated:** May 2026
