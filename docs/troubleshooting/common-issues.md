# Troubleshooting

## Common Issues

### Data Loading

**Error: "No such file or directory: merged_data/"**
- Solution: Download data first using `download.sh` or manually place OHLCV CSVs in `merged_data/`

**Error: "No metrics data found"**
- Solution: Metrics are optional. System runs with OHLCV signals only. To add metrics, download from Binance Vision and place in `merged_metrics/`

**Error: "Loaded 0 symbols"**
- Solution: Check CSV format. Must have columns: `open_time`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `taker_buy_volume`

### Signal Computation

**Error: "All signals are NaN"**
- Solution: Check data alignment. Ensure all symbols have same date range. Minimum 30 days of history required.

**Error: "IC is all NaN"**
- Solution: Check `MIN_SYMBOLS_PER_DATE`. If too high, many dates will have insufficient symbols. Reduce to 5-10.

### Composite Building

**Error: "composite_scores.csv is empty"**
- Solution: Check if signals have data. Run `ic_research.py` first to verify signals are computed.

**Error: "Composite IC is zero"**
- Solution: Check signal correlation. If signals are highly correlated, composite IC may be low. This is expected.

### Portfolio Simulation

**Error: "No P&L generated"**
- Solution: Check data alignment between composite scores and returns. Ensure dates match.

**Error: "NAV is flat"**
- Solution: Check composite scores. If all scores are near zero, positions will be small. Verify signal computation.

**Error: "Positions are all zero"**
- Solution: Check vol-scaling. If portfolio vol is very high, leverage will be near zero. Reduce `VOL_TARGET` or check for data outliers.

## Known Limitations

- **N=26 symbols** — Cross-sectional noise, but mitigated by T=1000 days
- **Survivorship bias** — Only active symbols included, excludes delisted coins
- **No market-cap control** — Residualization not applied
- **In-sample simulation** — Not a proper walk-forward backtest
- **Crypto-specific risks** — Exchange outages, liquidity gaps, regulatory changes not modeled

## Debugging Steps

1. **Check data:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('merged_data/BTCUSDT-1d-full.csv')
   print(df.head())
   print(df.info())
   "
   ```

2. **Check signals:**
   ```bash
   python3 -c "
   from ic_research import load_data, compute_signals
   symbols = load_data()
   signals = compute_signals(symbols)
   for sig, df in signals.items():
       print(f'{sig}: {df.notna().sum().sum()} non-null values')
   "
   ```

3. **Check IC:**
   ```bash
   python3 -c "
   import pandas as pd
   ic_df = pd.read_csv('ic_results.csv')
   print(ic_df[ic_df['horizon']==1].sort_values('t_stat', ascending=False).head(10))
   "
   ```

4. **Check composite:**
   ```bash
   python3 -c "
   import pandas as pd
   comp = pd.read_csv('composite/composite_scores.csv')
   print(comp.head())
   print(comp.describe())
   "
   ```

5. **Check portfolio:**
   ```bash
   python3 -c "
   import pandas as pd
   pnl = pd.read_csv('portfolio/portfolio_pnl.csv')
   print(pnl[['date','nav','daily_ret','drawdown']].head(20))
   "
   ```

## Performance Optimization

- **Reduce symbols:** Test with subset (e.g., top 10 by volume)
- **Reduce history:** Test with recent data only (e.g., 2024-2025)
- **Reduce horizons:** Test fewer horizons (e.g., [1, 5, 20])
- **Increase MIN_SYMBOLS_PER_DATE:** Reduces computation time

**Status:** Production
