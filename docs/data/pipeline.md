# Data Pipeline

## Binance Vision Integration

### OHLCV Data

**Source:** `https://data.binance.vision/data/futures/um/daily/klines/{SYMBOL}/`

**Format:** Daily klines (1d interval)

**Columns:**
- `open_time` — Unix timestamp (ms)
- `open`, `high`, `low`, `close` — OHLC prices
- `volume` — Base asset volume
- `quote_volume` — Quote asset volume
- `taker_buy_volume` — Taker buy volume
- `taker_buy_quote_volume` — Taker buy quote volume

**Processing:**
1. Download ZIP files per symbol
2. Extract and merge into single CSV per symbol
3. Store in `merged_data/{SYMBOL}-1d-full.csv`
4. Parse `open_time` as datetime, normalize to date-only

### Metrics Data (Optional)

**Source:** `https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/`

**Format:** 5-minute aggregated metrics

**Columns:**
- `create_time` — 5-minute intervals
- `sum_open_interest` — OI in coin units
- `sum_open_interest_value` — OI in USD
- `count_long_short_ratio` — All users L/S ratio
- `sum_toptrader_long_short_ratio` — Top trader L/S ratio
- `sum_taker_long_short_vol_ratio` — Taker L/S volume ratio

**Processing:**
1. Download ZIP files per symbol
2. Aggregate 5-min to daily (LAST for OI, MEAN for ratios)
3. Store in `merged_metrics/{SYMBOL}-metrics-daily.csv`
4. Align with OHLCV dates

## Data Alignment

- **Index:** Date-only (normalize timestamps to midnight UTC)
- **Union:** Use union of all available dates across symbols
- **Missing:** Forward-fill or drop rows with insufficient symbols
- **Min symbols per date:** 15 (configurable)

## Data Quality Checks

- **Duplicates:** Remove duplicate dates per symbol
- **Gaps:** Identify and log missing dates
- **Outliers:** Winsorize forward returns (1st-99th percentile)
- **NaN:** Drop rows with insufficient data

## File Structure

```
merged_data/
├── BTCUSDT-1d-full.csv
├── ETHUSDT-1d-full.csv
└── ... (26 symbols)

merged_metrics/
├── BTCUSDT-metrics-daily.csv
├── ETHUSDT-metrics-daily.csv
└── ... (optional)
```

**Status:** Production
