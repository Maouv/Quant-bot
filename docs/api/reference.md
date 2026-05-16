# API Reference

## Input Formats

### OHLCV Data (merged_data/*.csv)

```
open_time,open,high,low,close,volume,quote_volume,taker_buy_volume,taker_buy_quote_volume
1640995200000,47000.0,47500.0,46800.0,47200.0,1234.5,58234567.8,567.2,26789012.3
```

**Columns:**
- `open_time` — Unix timestamp (ms)
- `open`, `high`, `low`, `close` — OHLC prices (float)
- `volume` — Base asset volume (float)
- `quote_volume` — Quote asset volume (float)
- `taker_buy_volume` — Taker buy volume (float)
- `taker_buy_quote_volume` — Taker buy quote volume (float)

### Metrics Data (merged_metrics/*.csv)

```
date,sum_open_interest_value,count_long_short_ratio,sum_toptrader_long_short_ratio
2022-01-01,1234567890.5,1.23,1.15
```

**Columns:**
- `date` — Date (YYYY-MM-DD)
- `sum_open_interest_value` — OI in USD (float)
- `count_long_short_ratio` — All users L/S ratio (float)
- `sum_toptrader_long_short_ratio` — Top trader L/S ratio (float)

## Output Formats

### ic_results.csv

```
signal,horizon,IC_mean,IC_median,IC_std,ICIR,t_stat,t_stat_std,pct_positive,turnover_mean,IC_mean_in,IC_mean_out,IC_sub1_2022_2023,IC_sub2_2024_2025,n_days
volatility,1,0.0680,0.0712,0.3620,0.1870,7.0700,7.1500,0.5800,0.0234,0.0610,0.0840,0.0650,0.0710,1461
```

### composite_scores.csv

```
date,symbol,score_fast,score_slow,score_meta
2022-01-01,BTCUSDT,0.234,-0.156,0.089
2022-01-02,BTCUSDT,0.145,0.267,-0.034
```

### portfolio_pnl.csv

```
date,nav,daily_ret,gross_ret,cost_pct,drawdown,gross_exp,net_exp,n_long,n_short,port_vol_ann,btc_vol,regime_scale,paused
2022-01-01,100000.0,0.0000,0.0000,0.0000,0.0000,0.00,0.00,0,0,0.1500,0.6234,1.0000,False
2022-01-02,100234.0,0.0023,0.0031,-0.0008,-0.0023,1.45,0.12,8,3,0.1523,0.6145,1.0050,False
```

### portfolio_positions.csv

```
date,BTCUSDT,ETHUSDT,BNBUSDT,...
2022-01-01,0.0000,0.0000,0.0000,...
2022-01-02,0.0234,-0.0156,0.0089,...
```

## Configuration Parameters

All parameters defined at top of each script:

**ic_research.py**
- `REVERSAL_SHORT`, `REVERSAL_LONG` — Reversal windows (days)
- `MOMENTUM_MED`, `MOMENTUM_LONG` — Momentum windows (days)
- `VOL_WINDOW`, `VOL_SHORT` — Volatility windows (days)
- `MIN_SYMBOLS_PER_DATE` — Min symbols for IC calculation
- `IC_DECAY_HORIZONS` — Horizons to test
- `IN_SAMPLE_RATIO` — In-sample split ratio

**composite.py**
- `MIN_DAYS_WEIGHT` — Min days before signal gets weight
- `TSTAT_THRESHOLD` — t-stat threshold for inclusion
- `FAST_SIGNALS`, `FAST_WEIGHTS` — Fast composite definition
- `SLOW_SIGNALS`, `SLOW_WEIGHTS` — Slow composite definition

**portofolio.py**
- `VOL_TARGET` — Target annualized volatility
- `EWMA_HALFLIFE` — Vol EWMA halflife (days)
- `MAX_GROSS_EXPOSURE` — Gross exposure hard cap
- `MAX_DRAWDOWN_FLAT` — Flatten at this drawdown
- `DAILY_LOSS_LIMIT` — Pause at this daily loss
- `COST_PER_TRADE` — Transaction cost (bps)

**Status:** Production
