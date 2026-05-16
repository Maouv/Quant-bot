#!/usr/bin/env python3
"""
EXP6: Signal Decay Exit + Partial Take-Profit
==============================================
Experimental modification to portfolio.py testing:
1. Signal decay exit - force exit when conviction drops below threshold
2. Partial take-profit - reduce position when hitting profit target

Based on: portofolio.py
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
OUT_DIR       = "./experimental/exp6_signal_decay"

# Vol targeting
VOL_TARGET    = 0.15
EWMA_HALFLIFE = 60
VOL_EWMA_HL   = 30

# Position limits
MAX_POS_PER_SYMBOL = 0.12
MAX_SHORT_PER_SYMBOL = 0.06
MAX_GROSS_EXPOSURE = 3.0
MAX_NET_EXPOSURE   = 0.10

# Risk rules
DAILY_LOSS_LIMIT   = -0.045
MAX_DRAWDOWN_FLAT  = -0.20
PAUSE_DAYS         = 3
RESTART_SIZE       = 0.75

# Turnover buffer
TURNOVER_BUFFER    = 0.02
MAX_TURNOVER_LEG   = 0.40

# Transaction costs
COST_PER_TRADE     = 0.0007

# Regime filter
BTC_VOL_WINDOW     = 60

# Simulation
INITIAL_NAV        = 100_000

# ═════════════════════════════════════════════════════════════════════════════
# EXP6: NEW CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
DECAY_EXIT_RATIO = 0.4   # exit if current_score < 40% of entry_score
PARTIAL_TP_PCT   = 0.05  # take profit at 5% gain
PARTIAL_TP_SIZE  = 0.50  # reduce position by 50%

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
    return sym_vol.clip(lower=0.05)


def compute_target_weights(
    scores: pd.Series,
    sym_vol: pd.Series,
) -> pd.Series:
    """Compute target weights for one day."""
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
        w_final[shorts.index] = w_short

    return w_final


def apply_leverage(
    w: pd.Series,
    portfolio_vol: float,
    btc_vol: float,
    btc_vol_median: float,
) -> pd.Series:
    """Scale weights by vol-targeting leverage × regime scaler."""
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
    """Only trade if |target - current| > tau."""
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
    btc_vol_series: pd.Series,
    btc_vol_median: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main simulation loop with EXP6 modifications:
    1. Signal decay exit
    2. Partial take-profit
    """
    print("\n=== EXP6: SIGNAL DECAY EXIT + PARTIAL TP ===")
    print(f"Decay exit ratio: {DECAY_EXIT_RATIO}")
    print(f"Partial TP: {PARTIAL_TP_PCT*100}% gain → reduce {PARTIAL_TP_SIZE*100}%")

    sym_vol = compute_symbol_vol(returns)

    all_dates = scores.index.intersection(returns.index)
    all_dates = all_dates[all_dates >= scores.index[0]]

    # State
    nav             = INITIAL_NAV
    hwm             = INITIAL_NAV
    current_w       = pd.Series(dtype=float)
    port_returns    = []
    pause_counter   = 0

    # ═════════════════════════════════════════════════════════════════════════════
    # EXP6: STATE VARIABLES FOR SIGNAL DECAY + PARTIAL TP
    # ═════════════════════════════════════════════════════════════════════════════
    entry_score     = {}   # {symbol: score saat masuk posisi}
    entry_price     = {}   # {symbol: price saat masuk}
    entry_direction = {}   # {symbol: 'long' or 'short'}
    
    # Track decay/TP events
    n_decay_exits   = 0
    n_partial_tps   = 0

    # Output storage
    pnl_rows      = []
    position_rows = []

    port_vol_ewm = VOL_TARGET / np.sqrt(365)

    for i, date in enumerate(all_dates[:-1]):
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
            actual_w = pd.Series(0.0, index=current_w.index)
            trades   = current_w.abs()
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

                # ═════════════════════════════════════════════════════════════════
                # EXP6: SIGNAL DECAY EXIT + PARTIAL TP (BEFORE TURNOVER BUFFER)
                # ═════════════════════════════════════════════════════════════════
                target_w_override = pd.Series(dtype=float)
                
                # Get close prices for today
                close_today = {}
                for sym in score_today.index:
                    if sym in returns.columns and date in returns.index:
                        # Need to reconstruct price from returns
                        # Use cumulative returns from a base
                        if sym in entry_price:
                            # Already have entry price, compute current from previous
                            prev_date = all_dates[i - 1] if i > 0 else date
                            if prev_date in returns.index and not pd.isna(returns.loc[prev_date, sym]):
                                close_today[sym] = entry_price[sym] * (1 + returns.loc[date, sym])
                            else:
                                close_today[sym] = entry_price[sym]
                        else:
                            # No entry yet, use a notional base of 1.0
                            close_today[sym] = 1.0
                    else:
                        close_today[sym] = entry_price.get(sym, 1.0)
                
                # Process each position
                for sym in list(current_w.index):
                    if abs(current_w.get(sym, 0)) < 0.001:
                        continue
                    
                    current_score = score_today.get(sym, 0)
                    
                    # Signal decay exit
                    if sym in entry_score:
                        if abs(current_score) < DECAY_EXIT_RATIO * abs(entry_score[sym]):
                            # Signal conviction dropped → force exit bypass buffer
                            target_w_override[sym] = 0
                            n_decay_exits += 1
                    
                    # Partial take-profit
                    if sym in entry_price and sym in entry_direction:
                        current_price = close_today.get(sym, entry_price[sym])
                        pnl_pct = (current_price / entry_price[sym]) - 1
                        if entry_direction[sym] == 'long' and pnl_pct > PARTIAL_TP_PCT:
                            current_w[sym] *= PARTIAL_TP_SIZE
                            n_partial_tps += 1
                        elif entry_direction[sym] == 'short' and pnl_pct < -PARTIAL_TP_PCT:
                            current_w[sym] *= PARTIAL_TP_SIZE
                            n_partial_tps += 1
                
                # Apply overrides to target weights
                for sym, override_val in target_w_override.items():
                    if sym in target_w_lev.index:
                        target_w_lev[sym] = override_val
                    else:
                        target_w_lev[sym] = override_val

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
            hwm = nav
            print(f"  🚨 [{date.date()}] Max DD {dd:.2%} → flatten + pause {PAUSE_DAYS}d")

        # ═════════════════════════════════════════════════════════════════════════════
        # EXP6: UPDATE ENTRY STATE WHEN POSITIONS CHANGE
        # ═════════════════════════════════════════════════════════════════════════════
        for sym in actual_w.index:
            if abs(actual_w[sym]) > 0.001 and sym not in entry_score:
                # New position opened
                entry_score[sym] = score_today.get(sym, 0)
                entry_price[sym] = close_today.get(sym, 1.0) if sym in close_today else 1.0
                entry_direction[sym] = 'long' if actual_w[sym] > 0 else 'short'
            elif abs(actual_w[sym]) < 0.001:
                # Position closed
                entry_score.pop(sym, None)
                entry_price.pop(sym, None)
                entry_direction.pop(sym, None)

        # ── 10. UPDATE STATE ───────────────────────────────────────────────
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

    n_days = len(pnl_df)
    print(f"\nSimulation complete: {n_days} days")
    print(f"  Decay exits: {n_decay_exits} ({n_decay_exits/n_days:.2f}/day avg)")
    print(f"  Partial TPs: {n_partial_tps} ({n_partial_tps/n_days:.2f}/day avg)")
    
    return position_df, pnl_df, n_decay_exits, n_partial_tps


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(pnl_df: pd.DataFrame, n_decay_exits: int, n_partial_tps: int) -> dict:
    """Compute all performance metrics from P&L DataFrame."""
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
    n_days       = len(rets)

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
        "n_days":          n_days,
        # EXP6 specific metrics
        "n_decay_exits":   n_decay_exits,
        "n_partial_tps":   n_partial_tps,
        "avg_decay_exits_per_day": n_decay_exits / n_days if n_days > 0 else 0,
        "avg_partial_tp_per_day": n_partial_tps / n_days if n_days > 0 else 0,
    }


def compute_monthly_returns(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly return table."""
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
        f"EXP6: Signal Decay Exit + Partial TP\n"
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

    with open(f"{OUT_DIR}/portfolio_metrics.txt", "w") as f:
        f.write("=== EXP6: SIGNAL DECAY EXIT + PARTIAL TP ===\n")
        f.write("(Experimental modification to portfolio.py)\n\n")
        
        f.write("--- EXP6 Parameters ---\n")
        f.write(f"Decay exit ratio: {DECAY_EXIT_RATIO}\n")
        f.write(f"Partial TP: {PARTIAL_TP_PCT*100}% gain → reduce {PARTIAL_TP_SIZE*100}%\n\n")

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

        f.write("--- EXP6 Specific Metrics ---\n")
        f.write(f"Total Decay Exits:  {metrics['n_decay_exits']:>10}\n")
        f.write(f"Total Partial TPs:  {metrics['n_partial_tps']:>10}\n")
        f.write(f"Avg Decay Exits/Day:{metrics['avg_decay_exits_per_day']:>10.3f}\n")
        f.write(f"Avg Partial TP/Day: {metrics['avg_partial_tp_per_day']:>10.3f}\n\n")

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

    scores  = load_meta_scores()
    returns = load_returns()

    common_syms = scores.columns.intersection(returns.columns)
    scores  = scores[common_syms]
    returns = returns[common_syms]
    print(f"Common symbols: {len(common_syms)}")

    btc_vol_series, btc_vol_median = load_btc_vol(returns)

    positions, pnl_df, n_decay_exits, n_partial_tps = run_simulation(
        scores, returns, btc_vol_series, btc_vol_median
    )

    if pnl_df.empty:
        print("❌ No P&L generated — check data alignment")
        return

    print("\n=== COMPUTING METRICS ===")
    metrics     = compute_metrics(pnl_df, n_decay_exits, n_partial_tps)
    monthly     = compute_monthly_returns(pnl_df)
    attribution = compute_attribution(pnl_df, positions, returns)

    print(f"\n  CAGR:         {metrics['cagr']:>8.2%}")
    print(f"  Sharpe:       {metrics['sharpe']:>8.2f}")
    print(f"  Sortino:      {metrics['sortino']:>8.2f}")
    print(f"  Max DD:       {metrics['max_drawdown']:>8.2%}")
    print(f"  Calmar:       {metrics['calmar']:>8.2f}")
    print(f"  Ann Vol:      {metrics['ann_vol']:>8.2%}")
    print(f"  Avg Gross:    {metrics['avg_gross_exp']:>8.2f}x")
    print(f"  Total Cost:   {metrics['total_cost_pct']:>8.2%}")
    print(f"  Decay Exits:  {metrics['n_decay_exits']:>8}")
    print(f"  Partial TPs:  {metrics['n_partial_tps']:>8}")

    create_charts(pnl_df, metrics)
    save_outputs(pnl_df, positions, metrics, monthly, attribution)

    print("\nDone. Read experimental/exp6_signal_decay/portfolio_metrics.txt for full results.")


if __name__ == "__main__":
    main()
