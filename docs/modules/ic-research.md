# IC Research Module

Discovers and validates alpha signals through cross-sectional IC analysis.

## Signals (14 total)

**OHLCV (11)**
- `reversal_1d`, `reversal_1w` — Price reversals
- `momentum_30d`, `momentum_90d` — Momentum
- `volatility` — 30-day realized vol (negated)
- `liquidity`, `quote_liquidity` — Liquidity proxies
- `volume_compression`, `vol_compression` — Compression ratios
- `taker_buy_contrarian` — Taker buy ratio (negated)
- `price_to_high` — Proximity to 30-day high

**Metrics (3, optional)**
- `ls_contrarian` — All users L/S ratio (negated)
- `top_trader_contrarian` — Top trader L/S ratio (negated)
- `oi_price_signal` — OI-price divergence (negated)

## Process

1. **Compute signals** — Per symbol, per day
2. **Preprocess** — Z-score normalize (cross-sectional), clip to [-3,3], winsorize returns
3. **Calculate IC** — Spearman rank correlation (daily, cross-sectional)
4. **Aggregate** — IC_mean, IC_std, ICIR, Newey-West t-stat (5 lags)
5. **Validate** — Subsample stability (2022-2023 vs 2024-2025), in-sample vs out-of-sample

## Outputs

- `ic_results.csv` — IC metrics per signal per horizon
- `ic_chart.png` — IC at h=1
- `ic_subsample_chart.png` — Subsample stability
- `ic_decay_chart.png` — IC decay across horizons
- `ic_summary.txt` — Text summary

## Configuration

```python
REVERSAL_SHORT = 1              # 1-day reversal
REVERSAL_LONG = 7               # 1-week reversal
MOMENTUM_MED = 30               # 30-day momentum
VOL_WINDOW = 30                 # 30-day volatility
MIN_SYMBOLS_PER_DATE = 15       # Min symbols for IC
IC_DECAY_HORIZONS = [1,2,3,5,10,20]
IN_SAMPLE_RATIO = 0.7           # 70% in-sample
```

## Interpretation

- **t-stat > 2.0** — Statistically significant
- **ICIR > 0.5** — Strong signal
- **% IC > 0 > 55%** — Consistent signal
- **Sign flip (subsample)** — Overfitting indicator
- **IC degradation (OOS)** — Overfitting indicator

**Status:** Production
