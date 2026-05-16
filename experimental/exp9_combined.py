#!/usr/bin/env python3
"""
EXP9: BTC Gate + Dual Regime (Combined)
=======================================
Combines:
- BTC momentum gate from exp8: reduce short exposure when BTC rallying
- Dual regime filter from exp5: fast/slow vol scaling for all exposure

BTC gate → scale short leg only
Regime scale → scale all exposure (long + short)
Both applied independently.
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
OUT_DIR       = "./experimental/exp9_combined"

# Vol targeting
VOL_TARGET    = 0.15
EWMA_HALFLIFE = 60
VOL_EWMA_HL   = 30

# Position limits (from exp8)
MAX_POS_PER_SYMBOL   = 0.13
MAX_SHORT_PER_SYMBOL = 0.05
MAX_GROSS_EXPOSURE   = 3.0
MAX_NET_EXPOSURE     = 0.10

# Risk rules (from exp8)
DAILY_LOSS_LIMIT   = -0.035
MAX_DRAWDOWN_FLAT  = -0.17
PAUSE_DAYS         = 4
RESTART_SIZE       = 0.75

# Turnover buffer
TURNOVER_BUFFER    = 0.02
MAX_TURNOVER_LEG   = 0.40

# Transaction costs
COST_PER_TRADE     = 0.0007

# ── EXP9: BTC GATE + DUAL REGIME ───────────────────────────────────────────────
# BTC momentum gate (from exp8)
BTC_MOM_WINDOW    = 20
BTC_MOM_THRESHOLD = 0.15
SHORT_SCALE_GATE  = 0.30

# Dual regime filter (from exp5)
BTC_VOL_FAST = 10
BTC_VOL_SLOW = 60

# Simulation
INITIAL_NAV        = 100_000

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_meta_scores() -> pd.DataFrame:
    fp = f"{COMPOSITE_DIR}/composite_scores_meta.csv"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Meta scores not found: {fp}")
    df = pd.read_csv(fp, index_col="date", parse_dates=True)
    df.index = df.index.normalize()
    print(f"Loaded meta scores: {len(df)} days × {len(df.columns)} symbols")
    return df


def load_returns() -> pd.DataFrame:
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


def load_btc_vol(returns: pd.DataFrame) -> tuple:
    """Dual-window BTC vol for regime filter."""
    if "BTCUSDT" not in returns.columns:
        print("  ⚠️  BTCUSDT not found — regime filter disabled")
        dummy = pd.Series(1.0, index=returns.index)
        return dummy, dummy, 1.0, 1.0
    
    btc = returns["BTCUSDT"]
    
    btc_vol_fast = btc.rolling(BTC_VOL_FAST).std() * np.sqrt(365)
    btc_vol_fast_median = btc_vol_fast.median()
    
    btc_vol_slow = btc.rolling(BTC_VOL_SLOW).std() * np.sqrt(365)
    btc_vol_slow_median = btc_vol_slow.median()
    
    print(f"BTC {BTC_VOL_FAST}d vol median: {btc_vol_fast_median:.1%}")
    print(f"BTC {BTC_VOL_SLOW}d vol median: {btc_vol_slow_median:.1%}")
    
    return btc_vol_fast, btc_vol_slow, btc_vol_fast_median, btc_vol_slow_median


def load_btc_close() -> pd.Series:
    """Load BTC close prices for momentum calculation."""
    fp = f"{OHLCV_DIR}/BTCUSDT-1d-full.csv"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"BTC data not found: {fp}")
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("date").sort_index()
    df.index = df.index.normalize()
    return df["close"].astype(float)


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def compute_symbol_vol(returns: pd.DataFrame) -> pd.DataFrame:
    alpha = 1 - np.exp(-np.log(2) / VOL_EWMA_HL)
    sym_vol = returns.ewm(alpha=alpha, min_periods=10).std() * np.sqrt(365)
    sym_vol = sym_vol.replace(0, np.nan).ffill().fillna(0.5)
    return sym_vol.clip(lower=0.05)


def compute_target_weights(
    scores: pd.Series,
    sym_vol: pd.Series,
    short_scale: float = 1.0,
) -> pd.Series:
    """Compute target weights. short_scale from BTC gate."""
    common = scores.index.intersection(sym_vol.index)
    s = scores[common].dropna()
    v = sym_vol[common].reindex(s.index).fillna(0.5)

    if len(s) < 5:
        return pd.Series(dtype=float)

    w_raw = s / v
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan).dropna()

    if len(w_raw) == 0:
        return pd.Series(dtype=float)

    longs  = w_raw[w_raw > 0]
    shorts = w_raw[w_raw < 0]

    w_final = pd.Series(0.0, index=w_raw.index)

    if len(longs) > 0:
        w_long = longs / longs.sum()
        w_long = w_long.clip(upper=MAX_POS_PER_SYMBOL)
        w_long = w_long / w_long.sum()
        w_final[longs.index] = w_long

    if len(shorts) > 0:
        w_short = shorts / shorts.abs().sum()
        w_short = w_short.clip(lower=-MAX_SHORT_PER_SYMBOL)
        w_short = w_short / w_short.abs().sum()
        w_final[shorts.index] = w_short * short_scale

    return w_final


def apply_leverage(
    w: pd.Series,
    portfolio_vol: float,
    btc_vol_fast: float,
    btc_vol_slow: float,
    btc_vol_fast_median: float,
    btc_vol_slow_median: float,
) -> tuple:
    """
    Apply vol-targeting leverage + dual regime scaling.
    Returns (scaled_weights, fast_scale, slow_scale, regime_scale)
    """
    if portfolio_vol <= 0:
        lev = 1.0
    else:
        lev = min(VOL_TARGET / portfolio_vol, MAX_GROSS_EXPOSURE / 2)

    # Dual regime: fast trigger + slow scaler
    fast_trigger = btc_vol_fast > (2.0 * btc_vol_fast_median)
    fast_scale = 0.5 if fast_trigger else 1.0
    slow_scale = min(1.0, btc_vol_slow_median / btc_vol_slow) if btc_vol_slow > 0 else 1.0
    regime_scale = fast_scale * slow_scale

    lev = lev * regime_scale
    return w * lev, fast_scale, slow_scale, regime_scale


def apply_turnover_buffer(
    target_w: pd.Series,
    current_w: pd.Series,
    tau: float = TURNOVER_BUFFER,
) -> pd.Series:
    all_syms = target_w.index.union(current_w.index)
    t = target_w.reindex(all_syms, fill_value=0.0)
    c = current_w.reindex(all_syms, fill_value=0.0)

    delta = t - c
    trade_mask = delta.abs() > tau
    actual_w = c.copy()
    actual_w[trade_mask] = t[trade_mask]

    long_delta  = (actual_w - c).clip(lower=0)
    short_delta = (actual_w - c).clip(upper=0).abs()
    if long_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / long_delta.sum()
        actual_w[long_delta > 0] = c[long_delta > 0] + long_delta[long_delta > 0] * scale
    if short_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / short_delta.sum()
        actual_w[short_delta > 0] = c[short_delta > 0] - short_delta[short_delta > 0] * scale

    net = actual_w.sum()
    if abs(net) > MAX_NET_EXPOSURE:
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
    btc_vol_fast_series: pd.Series,
    btc_vol_slow_series: pd.Series,
    btc_vol_fast_median: float,
    btc_vol_slow_median: float,
    btc_mom: pd.Series,
) -> tuple:
    print("\n=== EXP9: BTC GATE + DUAL REGIME (COMBINED) ===")
    print(f"BTC gate: {BTC_MOM_WINDOW}d momentum > {BTC_MOM_THRESHOLD:.0%} → shorts {SHORT_SCALE_GATE:.0%}")
    print(f"Regime: fast={BTC_VOL_FAST}d, slow={BTC_VOL_SLOW}d dual window\n")

    sym_vol = compute_symbol_vol(returns)

    all_dates = scores.index.intersection(returns.index)
    all_dates = all_dates[all_dates >= scores.index[0]]

    nav             = INITIAL_NAV
    hwm             = INITIAL_NAV
    current_w       = pd.Series(dtype=float)
    port_returns    = []
    pause_counter   = 0

    pnl_rows      = []
    position_rows = []

    port_vol_ewm = VOL_TARGET / np.sqrt(365)

    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]

        if date not in scores.index:
            continue
        score_today = scores.loc[date].dropna()
        if len(score_today) < 5:
            continue

        # Get dual-window vol values
        btc_vol_fast_today = btc_vol_fast_series.get(date, btc_vol_fast_median)
        btc_vol_slow_today = btc_vol_slow_series.get(date, btc_vol_slow_median)
        if pd.isna(btc_vol_fast_today) or btc_vol_fast_today == 0:
            btc_vol_fast_today = btc_vol_fast_median
        if pd.isna(btc_vol_slow_today) or btc_vol_slow_today == 0:
            btc_vol_slow_today = btc_vol_slow_median

        # BTC momentum gate (independent from regime)
        btc_mom_today = btc_mom.get(date, 0)
        if pd.isna(btc_mom_today):
            btc_mom_today = 0
        short_scale = SHORT_SCALE_GATE if btc_mom_today > BTC_MOM_THRESHOLD else 1.0
        gate_active = short_scale < 1.0

        if pause_counter > 0:
            pause_counter -= 1
            actual_w = pd.Series(0.0, index=current_w.index)
            trades   = current_w.abs()
            current_w = pd.Series(dtype=float)
            fast_scale, slow_scale, regime_scale = 1.0, 1.0, 1.0
        else:
            sym_vol_today = sym_vol.loc[date] if date in sym_vol.index else pd.Series(0.5, index=score_today.index)
            target_w_raw  = compute_target_weights(score_today, sym_vol_today, short_scale)

            if len(target_w_raw) == 0:
                actual_w = pd.Series(dtype=float)
                trades   = pd.Series(dtype=float)
                fast_scale, slow_scale, regime_scale = 1.0, 1.0, 1.0
            else:
                target_w_lev, fast_scale, slow_scale, regime_scale = apply_leverage(
                    target_w_raw, port_vol_ewm * np.sqrt(365),
                    btc_vol_fast_today, btc_vol_slow_today,
                    btc_vol_fast_median, btc_vol_slow_median
                )

                actual_w = apply_turnover_buffer(target_w_lev, current_w)
                trades   = (actual_w - current_w.reindex(actual_w.index, fill_value=0)).abs()

        cost_pct = trades.sum() * COST_PER_TRADE

        if next_date in returns.index and len(actual_w) > 0:
            ret_next   = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            gross_ret  = (actual_w * ret_next).sum()
        else:
            gross_ret = 0.0

        net_ret = gross_ret - cost_pct
        nav     = nav * (1 + net_ret)
        hwm     = max(hwm, nav)
        dd      = (nav - hwm) / hwm

        alpha_port   = 1 - np.exp(-np.log(2) / EWMA_HALFLIFE)
        port_vol_ewm = np.sqrt(
            alpha_port * net_ret**2 + (1 - alpha_port) * port_vol_ewm**2
        )

        if dd < MAX_DRAWDOWN_FLAT and pause_counter == 0:
            pause_counter = PAUSE_DAYS
            hwm = nav
            print(f"  🚨 [{date.date()}] Max DD {dd:.2%} → flatten + pause {PAUSE_DAYS}d")

        if len(actual_w) > 0 and next_date in returns.index:
            ret_drift  = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            current_w  = actual_w * (1 + ret_drift)
        else:
            current_w = actual_w

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
            "btc_vol_fast": btc_vol_fast_today,
            "btc_vol_slow": btc_vol_slow_today,
            "fast_scale":   fast_scale,
            "slow_scale":   slow_scale,
            "regime_scale": regime_scale,
            "paused":       pause_counter > 0,
            "btc_mom":      btc_mom_today,
            "short_scale":  short_scale,
            "gate_active":  gate_active,
        })

        if len(actual_w) > 0:
            w_row = actual_w.to_dict()
            w_row["date"] = date
            position_rows.append(w_row)

        if i % 100 == 0:
            print(f"  [{date.date()}] NAV={nav:,.0f}  DD={dd:.1%}  "
                  f"Gross={gross_exp:.1f}x  n={n_long}L/{n_short}S  "
                  f"BTC_mom={btc_mom_today:+.1%}  gate={'ON' if gate_active else 'OFF'}")

    pnl_df      = pd.DataFrame(pnl_rows).set_index("date")
    position_df = pd.DataFrame(position_rows).set_index("date") if position_rows else pd.DataFrame()

    print(f"\nSimulation complete: {len(pnl_df)} days")
    return position_df, pnl_df


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(pnl_df: pd.DataFrame) -> dict:
    rets = pnl_df["daily_ret"].dropna()
    nav  = pnl_df["nav"]

    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    n_years   = len(rets) / 365
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol   = rets.std() * np.sqrt(365)
    sharpe    = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0
    downside  = rets[rets < 0].std() * np.sqrt(365)
    sortino   = (cagr / downside) if downside > 0 else 0
    max_dd    = pnl_df["drawdown"].min()
    calmar    = cagr / abs(max_dd) if max_dd < 0 else 0

    total_cost = pnl_df["cost_pct"].sum()
    avg_daily_cost = pnl_df["cost_pct"].mean()
    avg_turnover = avg_daily_cost / COST_PER_TRADE

    win_rate  = (rets > 0).mean()
    avg_win   = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss  = rets[rets < 0].mean() if (rets < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    avg_gross = pnl_df["gross_exp"].mean()
    avg_net   = pnl_df["net_exp"].mean()

    n_daily_loss = (rets < DAILY_LOSS_LIMIT).sum()
    n_paused     = pnl_df["paused"].sum() if "paused" in pnl_df else 0
    n_fast_triggers = (pnl_df["fast_scale"] < 1.0).sum() if "fast_scale" in pnl_df else 0

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
        "n_fast_triggers": n_fast_triggers,
        "n_days":          len(rets),
    }


def compute_monthly_returns(pnl_df: pd.DataFrame) -> pd.DataFrame:
    monthly = pnl_df["daily_ret"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
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
    print("Creating charts...")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    # NAV curve
    ax1 = fig.add_subplot(gs[0])
    nav_norm = pnl_df["nav"] / pnl_df["nav"].iloc[0]
    ax1.plot(nav_norm, color="steelblue", linewidth=1.5, label="Portfolio NAV")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("NAV (normalized)")
    ax1.set_title(
        f"EXP9: BTC Gate + Dual Regime (Combined)\n"
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
    ax3.axhline(MAX_GROSS_EXPOSURE, color="red", linewidth=0.8, linestyle="--",
                label=f"Max gross ({MAX_GROSS_EXPOSURE}x)")
    ax3.set_ylabel("Gross Exposure (×)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # BTC Momentum + Gate
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(pnl_df["btc_mom"], color="orange", linewidth=1, label="BTC 20d momentum")
    ax4.axhline(BTC_MOM_THRESHOLD, color="green", linewidth=1.5, linestyle="--",
                label=f"Gate threshold ({BTC_MOM_THRESHOLD:.0%})")
    ax4.axhline(0, color="black", linewidth=0.5)
    gate_mask = pnl_df["gate_active"]
    if gate_mask.any():
        ax4.fill_between(pnl_df.index, -0.5, 1.0, where=gate_mask, 
                         color="green", alpha=0.2, label="Gate active")
    ax4.set_ylabel("BTC Momentum")
    ax4.legend(fontsize=9, loc="upper left")
    ax4.grid(alpha=0.3)

    plt.savefig(f"{OUT_DIR}/portfolio_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/portfolio_chart.png")


# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────

def save_outputs(pnl_df, positions, metrics, monthly, attribution):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    pnl_df.to_csv(f"{OUT_DIR}/portfolio_pnl.csv")
    if not positions.empty:
        positions.to_csv(f"{OUT_DIR}/portfolio_positions.csv")
    if not attribution.empty:
        attribution.to_csv(f"{OUT_DIR}/portfolio_attribution.csv")
    monthly.to_csv(f"{OUT_DIR}/portfolio_monthly.csv")

    # Gate stats
    n_gate_days = pnl_df["gate_active"].sum()
    pct_gate = n_gate_days / len(pnl_df)
    
    # Fast trigger stats
    n_fast_days = (pnl_df["fast_scale"] < 1.0).sum()
    pct_fast = n_fast_days / len(pnl_df)

    with open(f"{OUT_DIR}/portfolio_metrics.txt", "w") as f:
        f.write("=== EXP9: BTC GATE + DUAL REGIME (COMBINED) ===\n")
        f.write("(In-sample simulation — not a proper walk-forward backtest)\n\n")

        f.write("--- Combined Parameters ---\n")
        f.write(f"BTC gate: {BTC_MOM_WINDOW}d momentum > {BTC_MOM_THRESHOLD:.0%} → shorts {SHORT_SCALE_GATE:.0%}\n")
        f.write(f"Regime: fast={BTC_VOL_FAST}d, slow={BTC_VOL_SLOW}d dual window\n\n")

        f.write("--- Trigger Statistics ---\n")
        f.write(f"Gate triggered:    {n_gate_days} days ({pct_gate:.1%})\n")
        f.write(f"Fast regime triggered: {n_fast_days} days ({pct_fast:.1%})\n\n")

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
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    scores  = load_meta_scores()
    returns = load_returns()

    # Align symbols
    common_syms = scores.columns.intersection(returns.columns)
    scores  = scores[common_syms]
    returns = returns[common_syms]
    print(f"Common symbols: {len(common_syms)}")

    # BTC dual-window vol for regime filter
    btc_vol_fast_series, btc_vol_slow_series, btc_vol_fast_median, btc_vol_slow_median = load_btc_vol(returns)

    # BTC momentum for gate
    btc_close = load_btc_close()
    btc_mom = btc_close / btc_close.shift(BTC_MOM_WINDOW) - 1
    print(f"BTC {BTC_MOM_WINDOW}d momentum: {btc_mom.dropna().min():.1%} to {btc_mom.dropna().max():.1%}")

    # Run simulation
    positions, pnl_df = run_simulation(
        scores, returns,
        btc_vol_fast_series, btc_vol_slow_series,
        btc_vol_fast_median, btc_vol_slow_median,
        btc_mom
    )

    if pnl_df.empty:
        print("❌ No P&L generated — check data alignment")
        return

    # Metrics
    print("\n=== COMPUTING METRICS ===")
    metrics     = compute_metrics(pnl_df)
    monthly     = compute_monthly_returns(pnl_df)
    attribution = compute_attribution(pnl_df, positions, returns)

    # Print key metrics
    n_gate_days = pnl_df["gate_active"].sum()
    pct_gate = n_gate_days / len(pnl_df)
    n_fast_days = (pnl_df["fast_scale"] < 1.0).sum()
    pct_fast = n_fast_days / len(pnl_df)
    
    print(f"\n  CAGR:         {metrics['cagr']:>8.2%}")
    print(f"  Sharpe:       {metrics['sharpe']:>8.2f}")
    print(f"  Sortino:      {metrics['sortino']:>8.2f}")
    print(f"  Max DD:       {metrics['max_drawdown']:>8.2%}")
    print(f"  Calmar:       {metrics['calmar']:>8.2f}")
    print(f"  Ann Vol:      {metrics['ann_vol']:>8.2%}")
    print(f"  Avg Gross:    {metrics['avg_gross_exp']:>8.2f}x")
    print(f"  Total Cost:   {metrics['total_cost_pct']:>8.2%}")
    print(f"\n  Gate triggered:    {n_gate_days} days ({pct_gate:.1%})")
    print(f"  Fast regime triggered: {n_fast_days} days ({pct_fast:.1%})")

    # Charts + save
    create_charts(pnl_df, metrics)
    save_outputs(pnl_df, positions, metrics, monthly, attribution)

    print("\nDone. Read experimental/exp9_combined/portfolio_metrics.txt for full results.")


if __name__ == "__main__":
    main()
