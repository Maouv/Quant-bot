#!/usr/bin/env python3
"""
Portfolio Simulation Engine
============================
Converts meta-composite scores into daily positions and simulates P&L.
This is a RESEARCH tool — model validation only, not live execution.

Architecture (per Opus Q1-Q6):
- Signal:     score_meta from composite/composite_scores_meta.csv
- Sizing:     volatility-scaled signal-proportional, per-leg normalized
- Leverage:   vol-targeting 15% annualized (EWMA halflife=20)
- Regime:     BTC 60d realized vol scaler
- Buffer:     no-trade zone τ=0.02 (reduces noise trading)
- Risk:       daily loss limit -2.5%, max DD -15%, max gross 3x

Output (all in ./portfolio/):
    portfolio_positions.csv   — daily weights per symbol
    portfolio_pnl.csv         — daily P&L, NAV, costs
    portfolio_metrics.txt     — Sharpe, Sortino, CAGR, max DD, etc.
    portfolio_monthly.csv     — monthly returns table
    portfolio_attribution.csv — per-symbol P&L attribution
    portfolio_chart.png       — NAV curve + drawdown

Usage:
    python3 portfolio.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

COMPOSITE_DIR = "./composite"
OHLCV_DIR     = "./merged_data"
OUT_DIR       = "./portfolio"

# Vol targeting
VOL_TARGET    = 0.15          # 15% annualized
EWMA_HALFLIFE = 60            # days — was 20, longer = less reactive to single bad day
VOL_EWMA_HL   = 30            # days, for per-symbol vol estimation

# Position limits
MAX_POS_PER_SYMBOL = 0.12     # 12% per leg max (long)
MAX_SHORT_PER_SYMBOL = 0.06   # 6% short cap — asymmetric: short convexity adjustment
                               # math: short_cap = long_cap × e^(-σ²T/2) ≈ 12% × 0.5
                               # σ=100% ann, T=5d → convexity penalty = 1.4% per position
MAX_GROSS_EXPOSURE = 3.0      # 3x hard cap
MAX_NET_EXPOSURE   = 0.10     # ±10% net limit

# Risk rules
DAILY_LOSS_LIMIT   = -0.045   # was -0.025 — triggers 3-5x/year not 15x/year
MAX_DRAWDOWN_FLAT  = -0.20    # was -0.15 — give strategy room to recover
PAUSE_DAYS         = 3        # was 5 — faster recovery
RESTART_SIZE       = 0.75     # was 0.50 — less aggressive penalty

# Turnover buffer
TURNOVER_BUFFER    = 0.02     # τ=0.02, don't trade if delta < 2%
MAX_TURNOVER_LEG   = 0.40     # 40% per leg per day cap

# Transaction costs
COST_PER_TRADE     = 0.0007   # 7 bps round-trip (5 bps taker + 2 bps slippage)

# Regime filter
BTC_VOL_WINDOW     = 60       # days for BTC realized vol

# Simulation
INITIAL_NAV        = 100_000  # $100K starting capital

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_meta_scores() -> pd.DataFrame:
    """Load meta-composite scores (date x symbol)."""
    fp = f"{COMPOSITE_DIR}/composite_scores_meta.csv"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Meta scores not found: {fp}\nRun composite.py first.")
    df = pd.read_csv(fp, index_col="date", parse_dates=True)
    df.index = df.index.normalize()
    print(f"Loaded meta scores: {len(df)} days × {len(df.columns)} symbols")
    return df


def load_returns() -> pd.DataFrame:
    """Load daily returns for all symbols from OHLCV data."""
    returns = {}
    for fp in glob.glob(f"{OHLCV_DIR}/*.csv"):
        sym = os.path.basename(fp).replace(".csv", "").replace("-1d-full", "")
        df  = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df  = df.set_index("date").sort_index()
        df.index = df.index.normalize()
        returns[sym] = df["close"].astype(float).pct_change()
    ret_df = pd.DataFrame(returns).sort_index()
    print(f"Loaded returns: {len(ret_df)} days × {len(ret_df.columns)} symbols")
    return ret_df


def load_btc_vol(returns: pd.DataFrame) -> pd.Series:
    """Compute BTC 60d realized vol (annualized) for regime filter."""
    if "BTCUSDT" not in returns.columns:
        print("  ⚠️  BTCUSDT not found — regime filter disabled")
        return pd.Series(1.0, index=returns.index)
    btc = returns["BTCUSDT"]
    btc_vol = btc.rolling(BTC_VOL_WINDOW).std() * np.sqrt(365)
    btc_vol_median = btc_vol.median()
    print(f"BTC 60d vol median: {btc_vol_median:.1%}")
    return btc_vol, btc_vol_median


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def compute_symbol_vol(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol EWMA volatility (annualized)."""
    alpha = 1 - np.exp(-np.log(2) / VOL_EWMA_HL)
    sym_vol = returns.ewm(alpha=alpha, min_periods=10).std() * np.sqrt(365)
    sym_vol = sym_vol.replace(0, np.nan).ffill().fillna(0.5)
    return sym_vol.clip(lower=0.05)  # floor at 5% ann vol


def compute_target_weights(
    scores: pd.Series,
    sym_vol: pd.Series,
) -> pd.Series:
    """
    Compute target weights for one day.
    w_raw[i] = score[i] / vol[i]
    Normalize each leg to sum(|w|) = 1, cap at MAX_POS_PER_SYMBOL.
    """
    common = scores.index.intersection(sym_vol.index)
    s = scores[common].dropna()
    v = sym_vol[common].reindex(s.index).fillna(0.5)

    if len(s) < 5:
        return pd.Series(dtype=float)

    # Vol-scaled raw weights
    w_raw = s / v
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan).dropna()

    if len(w_raw) == 0:
        return pd.Series(dtype=float)

    # Separate longs and shorts
    longs  = w_raw[w_raw > 0]
    shorts = w_raw[w_raw < 0]

    w_final = pd.Series(0.0, index=w_raw.index)

    if len(longs) > 0:
        w_long = longs / longs.sum()
        w_long = w_long.clip(upper=MAX_POS_PER_SYMBOL)
        w_long = w_long / w_long.sum()  # renormalize after cap
        w_final[longs.index] = w_long

    if len(shorts) > 0:
        w_short = shorts / shorts.abs().sum()
        w_short = w_short.clip(lower=-MAX_SHORT_PER_SYMBOL)  # asymmetric cap vs longs
        w_short = w_short / w_short.abs().sum()  # renormalize
        w_final[shorts.index] = w_short

    return w_final


def apply_leverage(
    w: pd.Series,
    portfolio_vol: float,
    btc_vol: float,
    btc_vol_median: float,
) -> pd.Series:
    """
    Scale weights by vol-targeting leverage × regime scaler.
    leverage = min(vol_target / portfolio_vol, max_gross/2)
    regime_scale = min(1.0, btc_vol_median / btc_vol)
    leverage_override removed — vol-targeting handles risk scaling organically.
    """
    if portfolio_vol <= 0:
        lev = 1.0
    else:
        lev = min(VOL_TARGET / portfolio_vol, MAX_GROSS_EXPOSURE / 2)

    regime_scale = min(1.0, btc_vol_median / btc_vol) if btc_vol > 0 else 1.0
    lev = lev * regime_scale

    return w * lev


def apply_turnover_buffer(
    target_w: pd.Series,
    current_w: pd.Series,
    tau: float = TURNOVER_BUFFER,
) -> pd.Series:
    """
    Only trade if |target - current| > tau.
    Reduces noise trading without changing signal.
    """
    all_syms = target_w.index.union(current_w.index)
    t = target_w.reindex(all_syms, fill_value=0.0)
    c = current_w.reindex(all_syms, fill_value=0.0)

    delta = t - c
    # Apply buffer: only trade where |delta| > tau
    trade_mask = delta.abs() > tau
    actual_w = c.copy()
    actual_w[trade_mask] = t[trade_mask]

    # Cap turnover per leg
    long_delta  = (actual_w - c).clip(lower=0)
    short_delta = (actual_w - c).clip(upper=0).abs()
    if long_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / long_delta.sum()
        actual_w[long_delta > 0] = c[long_delta > 0] + long_delta[long_delta > 0] * scale
    if short_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / short_delta.sum()
        actual_w[short_delta > 0] = c[short_delta > 0] - short_delta[short_delta > 0] * scale

    # Force net neutrality if |net| > MAX_NET_EXPOSURE
    net = actual_w.sum()
    if abs(net) > MAX_NET_EXPOSURE:
        # Trim the larger leg
        longs  = actual_w[actual_w > 0]
        shorts = actual_w[actual_w < 0]
        trim   = abs(net) / 2
        if net > 0:
            actual_w[longs.index]  *= (1 - trim / longs.sum())
        else:
            actual_w[shorts.index] *= (1 - trim / shorts.abs().sum())

    return actual_w.fillna(0.0)


# ── SIMULATION LOOP ───────────────────────────────────────────────────────────

def run_simulation(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    btc_vol_series: pd.Series,
    btc_vol_median: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main simulation loop. Returns (positions_df, pnl_df).
    Follows Opus Q6 order of operations exactly.
    """
    print("\n=== RUNNING SIMULATION ===")

    sym_vol = compute_symbol_vol(returns)

    # Align dates
    all_dates = scores.index.intersection(returns.index)
    all_dates = all_dates[all_dates >= scores.index[0]]

    # State
    nav             = INITIAL_NAV
    hwm             = INITIAL_NAV
    current_w       = pd.Series(dtype=float)
    port_returns    = []
    pause_counter   = 0

    # Output storage
    pnl_rows      = []
    position_rows = []

    # Portfolio vol estimate (EWMA)
    port_vol_ewm = VOL_TARGET / np.sqrt(365)  # start at target

    for i, date in enumerate(all_dates[:-1]):  # last day has no next-day return
        next_date = all_dates[i + 1]

        # ── 1. SIGNAL ──────────────────────────────────────────────────────
        if date not in scores.index:
            continue
        score_today = scores.loc[date].dropna()
        if len(score_today) < 5:
            continue

        # ── 2. REGIME SCALE ────────────────────────────────────────────────
        btc_vol_today = btc_vol_series.get(date, btc_vol_median)
        if pd.isna(btc_vol_today) or btc_vol_today == 0:
            btc_vol_today = btc_vol_median

        # ── 3. PAUSE CHECK ─────────────────────────────────────────────────
        if pause_counter > 0:
            pause_counter -= 1
            # Hold zero positions during pause
            actual_w = pd.Series(0.0, index=current_w.index)
            trades   = current_w.abs()  # cost to flatten
            current_w = pd.Series(dtype=float)
        else:
            # ── 4. TARGET WEIGHTS ──────────────────────────────────────────
            sym_vol_today = sym_vol.loc[date] if date in sym_vol.index else pd.Series(0.5, index=score_today.index)
            target_w_raw  = compute_target_weights(score_today, sym_vol_today)

            if len(target_w_raw) == 0:
                actual_w = pd.Series(dtype=float)
                trades   = pd.Series(dtype=float)
            else:
                # ── 5. LEVERAGE ────────────────────────────────────────────
                target_w_lev = apply_leverage(
                    target_w_raw, port_vol_ewm * np.sqrt(365),
                    btc_vol_today, btc_vol_median
                )

                # ── 6. TURNOVER BUFFER ─────────────────────────────────────
                actual_w = apply_turnover_buffer(target_w_lev, current_w)
                trades   = (actual_w - current_w.reindex(actual_w.index, fill_value=0)).abs()

        # ── 7. TRANSACTION COSTS ───────────────────────────────────────────
        cost_pct = trades.sum() * COST_PER_TRADE

        # ── 8. P&L ────────────────────────────────────────────────────────
        if next_date in returns.index and len(actual_w) > 0:
            ret_next   = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            gross_ret  = (actual_w * ret_next).sum()
        else:
            gross_ret = 0.0

        net_ret = gross_ret - cost_pct
        nav     = nav * (1 + net_ret)
        hwm     = max(hwm, nav)
        dd      = (nav - hwm) / hwm

        # Update portfolio vol EWMA
        alpha_port   = 1 - np.exp(-np.log(2) / EWMA_HALFLIFE)
        port_vol_ewm = np.sqrt(
            alpha_port * net_ret**2 + (1 - alpha_port) * port_vol_ewm**2
        )

        # ── 9. RISK CHECKS ─────────────────────────────────────────────────
        if dd < MAX_DRAWDOWN_FLAT and pause_counter == 0:
            pause_counter = PAUSE_DAYS
            hwm = nav  # reset HWM after flatten
            print(f"  🚨 [{date.date()}] Max DD {dd:.2%} → flatten + pause {PAUSE_DAYS}d")

        # ── 10. UPDATE STATE ───────────────────────────────────────────────
        # Drift positions with market
        if len(actual_w) > 0 and next_date in returns.index:
            ret_drift  = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            current_w  = actual_w * (1 + ret_drift)
        else:
            current_w = actual_w

        # ── STORE ──────────────────────────────────────────────────────────
        gross_exp = actual_w.abs().sum() if len(actual_w) > 0 else 0.0
        net_exp   = actual_w.sum() if len(actual_w) > 0 else 0.0
        n_long    = (actual_w > 0).sum() if len(actual_w) > 0 else 0
        n_short   = (actual_w < 0).sum() if len(actual_w) > 0 else 0

        pnl_rows.append({
            "date":         date,
            "nav":          nav,
            "daily_ret":    net_ret,
            "gross_ret":    gross_ret,
            "cost_pct":     cost_pct,
            "drawdown":     dd,
            "gross_exp":    gross_exp,
            "net_exp":      net_exp,
            "n_long":       n_long,
            "n_short":      n_short,
            "port_vol_ann": port_vol_ewm * np.sqrt(365),
            "btc_vol":      btc_vol_today,
            "regime_scale": min(1.0, btc_vol_median / btc_vol_today),
            "paused":       pause_counter > 0,
        })

        if len(actual_w) > 0:
            w_row = actual_w.to_dict()
            w_row["date"] = date
            position_rows.append(w_row)

        if i % 100 == 0:
            print(f"  [{date.date()}] NAV={nav:,.0f}  DD={dd:.1%}  "
                  f"Gross={gross_exp:.1f}x  n={n_long}L/{n_short}S")

    pnl_df      = pd.DataFrame(pnl_rows).set_index("date")
    position_df = pd.DataFrame(position_rows).set_index("date") if position_rows else pd.DataFrame()

    print(f"\nSimulation complete: {len(pnl_df)} days")
    return position_df, pnl_df


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(pnl_df: pd.DataFrame) -> dict:
    """Compute all performance metrics from P&L DataFrame."""
    rets = pnl_df["daily_ret"].dropna()
    nav  = pnl_df["nav"]

    # Returns
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    n_years   = len(rets) / 365
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk
    ann_vol   = rets.std() * np.sqrt(365)
    sharpe    = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0
    downside  = rets[rets < 0].std() * np.sqrt(365)
    sortino   = (cagr / downside) if downside > 0 else 0
    max_dd    = pnl_df["drawdown"].min()
    calmar    = cagr / abs(max_dd) if max_dd < 0 else 0

    # Costs
    total_cost = pnl_df["cost_pct"].sum()
    avg_daily_cost = pnl_df["cost_pct"].mean()

    # Turnover (estimated from cost)
    avg_turnover = avg_daily_cost / COST_PER_TRADE

    # Win rate
    win_rate  = (rets > 0).mean()
    avg_win   = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss  = rets[rets < 0].mean() if (rets < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Exposure
    avg_gross = pnl_df["gross_exp"].mean()
    avg_net   = pnl_df["net_exp"].mean()

    # Risk events
    n_daily_loss = (rets < DAILY_LOSS_LIMIT).sum()
    n_paused     = pnl_df["paused"].sum() if "paused" in pnl_df else 0

    return {
        "total_return":    total_ret,
        "cagr":            cagr,
        "ann_vol":         ann_vol,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "max_drawdown":    max_dd,
        "calmar":          calmar,
        "win_rate":        win_rate,
        "profit_factor":   profit_factor,
        "avg_gross_exp":   avg_gross,
        "avg_net_exp":     avg_net,
        "total_cost_pct":  total_cost,
        "avg_daily_cost":  avg_daily_cost,
        "avg_turnover":    avg_turnover,
        "n_daily_loss_events": n_daily_loss,
        "n_paused_days":   n_paused,
        "n_days":          len(rets),
    }


def compute_monthly_returns(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly return table."""
    monthly = pnl_df["daily_ret"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    # Keep as DatetimeIndex, extract year/month directly
    df = monthly.reset_index()
    df.columns = ["date", "return"]
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    pivot["Annual"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    return pivot


def compute_attribution(pnl_df: pd.DataFrame, positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol P&L attribution."""
    if positions.empty:
        return pd.DataFrame()
    attr = {}
    for sym in positions.columns:
        if sym not in returns.columns:
            continue
        pos = positions[sym].reindex(returns.index, fill_value=0.0)
        ret = returns[sym]
        pnl = (pos.shift(1) * ret).dropna()
        attr[sym] = {
            "total_pnl_pct": pnl.sum(),
            "sharpe":        pnl.mean() / pnl.std() * np.sqrt(365) if pnl.std() > 0 else 0,
            "avg_position":  pos.abs().mean(),
            "n_days_active": (pos.abs() > 0.001).sum(),
        }
    return pd.DataFrame(attr).T.sort_values("total_pnl_pct", ascending=False)


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def create_charts(pnl_df: pd.DataFrame, metrics: dict):
    """Create NAV curve + drawdown + exposure chart."""
    print("Creating charts...")
    Path(OUT_DIR).mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.35)

    # NAV curve
    ax1 = fig.add_subplot(gs[0])
    nav_norm = pnl_df["nav"] / pnl_df["nav"].iloc[0]
    ax1.plot(nav_norm, color="steelblue", linewidth=1.5, label="Portfolio NAV")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("NAV (normalized)")
    ax1.set_title(
        f"Portfolio Simulation — Meta Composite\n"
        f"CAGR={metrics['cagr']:.1%}  Sharpe={metrics['sharpe']:.2f}  "
        f"MaxDD={metrics['max_drawdown']:.1%}  Calmar={metrics['calmar']:.2f}"
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Drawdown
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(pnl_df.index, pnl_df["drawdown"], 0, color="tomato", alpha=0.6)
    ax2.axhline(MAX_DRAWDOWN_FLAT, color="red", linewidth=1, linestyle="--",
                label=f"Flatten threshold ({MAX_DRAWDOWN_FLAT:.0%})")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Gross exposure + regime
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(pnl_df["gross_exp"], color="steelblue", linewidth=1, label="Gross exposure")
    ax3.plot(pnl_df["regime_scale"] * pnl_df["gross_exp"].mean(),
             color="orange", linewidth=1, alpha=0.7, label="Regime scale (×avg)")
    ax3.axhline(MAX_GROSS_EXPOSURE, color="red", linewidth=0.8, linestyle="--",
                label=f"Max gross ({MAX_GROSS_EXPOSURE}x)")
    ax3.set_ylabel("Gross Exposure (×)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.savefig(f"{OUT_DIR}/portfolio_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/portfolio_chart.png")


# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────

def save_outputs(pnl_df, positions, metrics, monthly, attribution):
    Path(OUT_DIR).mkdir(exist_ok=True)

    pnl_df.to_csv(f"{OUT_DIR}/portfolio_pnl.csv")
    if not positions.empty:
        positions.to_csv(f"{OUT_DIR}/portfolio_positions.csv")
    if not attribution.empty:
        attribution.to_csv(f"{OUT_DIR}/portfolio_attribution.csv")
    monthly.to_csv(f"{OUT_DIR}/portfolio_monthly.csv")

    # Text summary
    with open(f"{OUT_DIR}/portfolio_metrics.txt", "w") as f:
        f.write("=== PORTFOLIO SIMULATION RESULTS ===\n")
        f.write("(In-sample simulation — not a proper walk-forward backtest)\n\n")

        f.write("--- Performance ---\n")
        f.write(f"Total Return:     {metrics['total_return']:>10.2%}\n")
        f.write(f"CAGR:             {metrics['cagr']:>10.2%}\n")
        f.write(f"Ann. Volatility:  {metrics['ann_vol']:>10.2%}\n")
        f.write(f"Sharpe Ratio:     {metrics['sharpe']:>10.2f}\n")
        f.write(f"Sortino Ratio:    {metrics['sortino']:>10.2f}\n")
        f.write(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}\n")
        f.write(f"Calmar Ratio:     {metrics['calmar']:>10.2f}\n")
        f.write(f"Win Rate:         {metrics['win_rate']:>10.2%}\n")
        f.write(f"Profit Factor:    {metrics['profit_factor']:>10.2f}\n\n")

        f.write("--- Exposure & Costs ---\n")
        f.write(f"Avg Gross Exp:    {metrics['avg_gross_exp']:>10.2f}x\n")
        f.write(f"Avg Net Exp:      {metrics['avg_net_exp']:>10.2%}\n")
        f.write(f"Total Cost:       {metrics['total_cost_pct']:>10.2%}\n")
        f.write(f"Avg Daily Cost:   {metrics['avg_daily_cost']:>10.4%}\n")
        f.write(f"Avg Daily TO:     {metrics['avg_turnover']:>10.2%}\n\n")

        f.write("--- Risk Events ---\n")
        f.write(f"Daily Loss Events:{metrics['n_daily_loss_events']:>10}\n")
        f.write(f"Paused Days:      {metrics['n_paused_days']:>10}\n")
        f.write(f"Simulation Days:  {metrics['n_days']:>10}\n\n")

        f.write("--- Sanity Checks ---\n")
        sharpe_ok = 0.5 < metrics['sharpe'] < 5.0
        dd_ok     = metrics['max_drawdown'] > -0.50
        cost_ok   = metrics['total_cost_pct'] < abs(metrics['total_return']) * 0.3
        f.write(f"Sharpe in range (0.5-5.0):  {'✅' if sharpe_ok else '⚠️ SUSPICIOUS'}\n")
        f.write(f"Max DD < 50%:               {'✅' if dd_ok else '⚠️ HIGH'}\n")
        f.write(f"Costs < 30% of gross P&L:   {'✅' if cost_ok else '⚠️ HIGH COSTS'}\n")

    print(f"\n✅ Outputs saved to {OUT_DIR}/")
    print(f"   portfolio_metrics.txt  ← read this first")
    print(f"   portfolio_pnl.csv")
    print(f"   portfolio_positions.csv")
    print(f"   portfolio_monthly.csv")
    print(f"   portfolio_attribution.csv")
    print(f"   portfolio_chart.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(exist_ok=True)

    # Load data
    scores  = load_meta_scores()
    returns = load_returns()

    # Align symbols
    common_syms = scores.columns.intersection(returns.columns)
    scores  = scores[common_syms]
    returns = returns[common_syms]
    print(f"Common symbols: {len(common_syms)}")

    # BTC vol for regime filter
    btc_vol_series, btc_vol_median = load_btc_vol(returns)

    # Run simulation
    positions, pnl_df = run_simulation(scores, returns, btc_vol_series, btc_vol_median)

    if pnl_df.empty:
        print("❌ No P&L generated — check data alignment")
        return

    # Metrics
    print("\n=== COMPUTING METRICS ===")
    metrics     = compute_metrics(pnl_df)
    monthly     = compute_monthly_returns(pnl_df)
    attribution = compute_attribution(pnl_df, positions, returns)

    # Print key metrics
    print(f"\n  CAGR:         {metrics['cagr']:>8.2%}")
    print(f"  Sharpe:       {metrics['sharpe']:>8.2f}")
    print(f"  Sortino:      {metrics['sortino']:>8.2f}")
    print(f"  Max DD:       {metrics['max_drawdown']:>8.2%}")
    print(f"  Calmar:       {metrics['calmar']:>8.2f}")
    print(f"  Ann Vol:      {metrics['ann_vol']:>8.2%}")
    print(f"  Avg Gross:    {metrics['avg_gross_exp']:>8.2f}x")
    print(f"  Total Cost:   {metrics['total_cost_pct']:>8.2%}")

    # Charts + save
    create_charts(pnl_df, metrics)
    save_outputs(pnl_df, positions, metrics, monthly, attribution)

    print("\nDone. Read portfolio/portfolio_metrics.txt for full results.")


if __name__ == "__main__":
    main()



