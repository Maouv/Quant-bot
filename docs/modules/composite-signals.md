# Composite Signals Module

Combines validated signals into Fast/Slow/Meta composites using ICIR-proportional weights.

## Composites

**Fast (h=1, intraday mean-reversion)**
- Signals: volatility, liquidity, reversal_1w
- Weights: volatility +0.42, liquidity -0.36, reversal_1w +0.22

**Slow (h=20, swing trading)**
- Signals: volatility, liquidity, momentum_30d, taker_buy_contrarian
- Weights: volatility +0.38, liquidity -0.32, momentum_30d +0.18, taker_buy_contrarian -0.12

**Meta (blend)**
- Fast 36% + Slow 64% (ICIR-proportional)
- Slow smoothed with 5-day rolling mean

## Process

1. Load OHLCV + metrics data
2. Compute signals (same as ic_research)
3. Z-score normalize cross-sectionally
4. Build Fast composite (h=1)
5. Build Slow composite (h=20)
6. Blend into Meta composite
7. Compute IC for each composite
8. Visualize and save

## Outputs

- `composite/composite_scores.csv` — Combined Fast/Slow/Meta scores
- `composite/composite_scores_meta.csv` — Meta composite only
- `composite/composite_ic.csv` — IC of composites vs individuals
- `composite/composite_chart.png` — Composite vs individual IC
- `composite/composite_decay_chart.png` — IC decay
- `composite/composite_summary.txt` — Text summary

## Configuration

```python
MIN_DAYS_WEIGHT = 120           # Min days before weight
TSTAT_THRESHOLD = 1.0           # t-stat threshold
IC_DECAY_HORIZONS = [1,2,3,5,10,20]
MIN_SYMBOLS = 10                # Min symbols for IC
```

**Status:** Production
