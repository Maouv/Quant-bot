# Portfolio Module

Converts meta-composite scores into daily positions and simulates P&L with risk management.

**Note:** Research tool for model validation, not live execution.

## Position Sizing

1. **Vol-scaled:** `w_raw[i] = score[i] / vol[i]`
2. **Per-leg normalization:** Separate longs/shorts, normalize each
3. **Position caps:** 12% long, 6% short per symbol
4. **Gross cap:** 3x hard limit

## Leverage Targeting

- Target: 15% annualized volatility
- EWMA halflife: 60 days
- Regime filter: BTC 60d realized vol scaler
- Formula: `leverage = min(vol_target / port_vol, 3x/2) × regime_scale`

## Turnover Buffer

- No-trade zone: 2% (reduce noise trading)
- Max turnover per leg: 40% per day
- Net exposure limit: ±10%

## Risk Management

- **Daily loss limit:** -4.5% → pause 3 days
- **Max drawdown:** -20% → flatten + pause
- **Restart size:** 75% of pre-pause
- **Transaction costs:** 7 bps round-trip

## Outputs

- `portfolio/portfolio_pnl.csv` — Daily NAV, returns, costs, drawdown
- `portfolio/portfolio_positions.csv` — Daily weights per symbol
- `portfolio/portfolio_metrics.txt` — Summary statistics
- `portfolio/portfolio_monthly.csv` — Monthly returns
- `portfolio/portfolio_attribution.csv` — Per-symbol P&L
- `portfolio/portfolio_chart.png` — NAV curve, drawdown, exposure

## Configuration

```python
VOL_TARGET = 0.15               # 15% annualized vol
EWMA_HALFLIFE = 60              # Vol EWMA halflife
MAX_GROSS_EXPOSURE = 3.0        # 3x hard cap
MAX_DRAWDOWN_FLAT = -0.20       # Flatten at -20% DD
DAILY_LOSS_LIMIT = -0.045       # Pause at -4.5% loss
COST_PER_TRADE = 0.0007         # 7 bps round-trip
TURNOVER_BUFFER = 0.02          # 2% no-trade zone
```

**Status:** Development
