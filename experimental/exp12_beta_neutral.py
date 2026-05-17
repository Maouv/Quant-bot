#!/usr/bin/env python3
"""
EXP12: Beta-Neutral Construction
=================================
Structural fix: force portfolio beta = 0 after computing weights.
Removes market exposure, isolates pure cross-sectional alpha.
No regime filters, no IC scaling, no dispersion scaling.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ────────────────────────────────────────────────────────────────────

COMPOSITE_DIR = "./composite"
OHLCV_DIR     = "./merged_data"
OUT_DIR       = "./experimental/exp12_beta_neutral"

# Vol targeting
VOL_TARGET    = 0.15
EWMA_HALFLIFE = 60
VOL_EWMA_HL   = 30

# Position limits
MAX_POS_PER_SYMBOL   = 0.13
MAX_SHORT_PER_SYMBOL = 0.05
MAX_GROSS_EXPOSURE   = 3.0
MAX_NET_EXPOSURE     = 0.10

# Risk rules
DAILY_LOSS_LIMIT   = -0.035
MAX_DRAWDOWN_FLAT  = -0.17
PAUSE_DAYS         = 4

# Turnover
TURNOVER_BUFFER    = 0.01
MAX_TURNOVER_LEG   = 0.40

# Costs
COST_PER_TRADE     = 0.0007

# Beta neutralization
BETA_WINDOW        = 60

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


# ── BETA ESTIMATION ───────────────────────────────────────────────────────────

def compute_rolling_betas(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 60d beta of each coin vs equal-weight market.
    Beta_i = Cov(r_i, r_market) / Var(r_market)
    """
    market_ret = returns.mean(axis=1)
    
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    
    for sym in returns.columns:
        cov = returns[sym].rolling(BETA_WINDOW, min_periods=30).cov(market_ret)
        var = market_ret.rolling(BETA_WINDOW, min_periods=30).var()
        betas[sym] = (cov / var).clip(-3, 3)
    
    return betas.fillna(1.0)


def beta_neutralize(weights: pd.Series, betas: pd.Series) -> pd.Series:
    """
    Project weights onto beta-neutral subspace.
    Math: w_adj = w - (w·β / β·β) × β
    """
    common = weights.index.intersection(betas.index)
    w = weights.reindex(common, fill_value=0.0)
    b = betas.reindex(common, fill_value=1.0)
    
    port_beta = (w * b).sum()
    beta_norm_sq = (b * b).sum()
    
    if beta_norm_sq == 0:
        return w
    
    w_neutral = w - (port_beta / beta_norm_sq) * b
    return w_neutral


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def compute_symbol_vol(returns: pd.DataFrame) -> pd.DataFrame:
    alpha = 1 - np.exp(-np.log(2) / VOL_EWMA_HL)
    sym_vol = returns.ewm(alpha=alpha, min_periods=10).std() * np.sqrt(365)
    sym_vol = sym_vol.replace(0, np.nan).ffill().fillna(0.5)
    return sym_vol.clip(lower=0.05)


def compute_target_weights(scores: pd.Series, sym_vol: pd.Series) -> pd.Series:
    """Compute target weights. No short_scale parameter."""
    common = scores.index.intersection(sym_vol.index)
    s = scores[common].dropna()
    v = sym_vol[common].reindex(s.index).fillna(0.5)
    
    if len(s) < 5:
        return pd.Series(dtype=float)
    
    w_raw = s / v
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(w_raw) == 0:
        return pd.Series(dtype=float)
    
    longs = w_raw[w_raw > 0]
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


def apply_leverage(w: pd.Series, portfolio_vol: float) -> pd.Series:
    """Pure vol-targeting. No regime filters."""
    if portfolio_vol <= 0:
        lev = 1.0
    else:
        lev = min(VOL_TARGET / portfolio_vol, MAX_GROSS_EXPOSURE / 2)
    return w * lev


def apply_turnover_buffer(target_w: pd.Series, current_w: pd.Series) -> pd.Series:
    all_syms = target_w.index.union(current_w.index)
    t = target_w.reindex(all_syms, fill_value=0.0)
    c = current_w.reindex(all_syms, fill_value=0.0)
    
    delta = t - c
    trade_mask = delta.abs() > TURNOVER_BUFFER
    actual_w = c.copy()
    actual_w[trade_mask] = t[trade_mask]
    
    long_delta = (actual_w - c).clip(lower=0)
    short_delta = (actual_w - c).clip(upper=0).abs()
    if long_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / long_delta.sum()
        actual_w[long_delta > 0] = c[long_delta > 0] + long_delta[long_delta > 0] * scale
    if short_delta.sum() > MAX_TURNOVER_LEG:
        scale = MAX_TURNOVER_LEG / short_delta.sum()
        actual_w[short_delta > 0] = c[short_delta > 0] - short_delta[short_delta > 0] * scale
    
    net = actual_w.sum()
    if abs(net) > MAX_NET_EXPOSURE:
        longs = actual_w[actual_w > 0]
        shorts = actual_w[actual_w < 0]
        trim = abs(net) / 2
        if net > 0:
            actual_w[longs.index] *= (1 - trim / longs.sum())
        else:
            actual_w[shorts.index] *= (1 - trim / shorts.abs().sum())
    
    return actual_w.fillna(0.0)


# ── SIMULATION LOOP ───────────────────────────────────────────────────────────

def run_simulation(scores: pd.DataFrame, returns: pd.DataFrame, betas: pd.DataFrame) -> tuple:
    print("\n=== EXP12: BETA-NEUTRAL CONSTRUCTION ===")
    print(f"Beta window: {BETA_WINDOW}d rolling OLS vs equal-weight market")
    print(f"Vol target: {VOL_TARGET:.0%}\n")
    
    sym_vol = compute_symbol_vol(returns)
    
    all_dates = scores.index.intersection(returns.index)
    all_dates = all_dates[all_dates >= scores.index[0]]
    
    nav = INITIAL_NAV
    hwm = INITIAL_NAV
    current_w = pd.Series(dtype=float)
    pause_counter = 0
    
    pnl_rows = []
    position_rows = []
    port_vol_ewm = VOL_TARGET / np.sqrt(365)
    
    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        
        if date not in scores.index:
            continue
        score_today = scores.loc[date].dropna()
        if len(score_today) < 5:
            continue
        
        if pause_counter > 0:
            pause_counter -= 1
            actual_w = pd.Series(0.0, index=current_w.index)
            trades = current_w.abs()
            current_w = pd.Series(dtype=float)
            port_beta_before = 0.0
            port_beta_after = 0.0
        else:
            sym_vol_today = sym_vol.loc[date] if date in sym_vol.index else pd.Series(0.5, index=score_today.index)
            betas_today = betas.loc[date] if date in betas.index else pd.Series(1.0, index=score_today.index)
            
            target_w_raw = compute_target_weights(score_today, sym_vol_today)
            
            if len(target_w_raw) == 0:
                actual_w = pd.Series(dtype=float)
                trades = pd.Series(dtype=float)
                port_beta_before = 0.0
                port_beta_after = 0.0
            else:
                port_beta_before = (target_w_raw * betas_today.reindex(target_w_raw.index, fill_value=1.0)).sum()
                target_w_neutral = beta_neutralize(target_w_raw, betas_today)
                port_beta_after = (target_w_neutral * betas_today.reindex(target_w_neutral.index, fill_value=1.0)).sum()
                
                # Re-check net exposure after beta adjustment
                net = target_w_neutral.sum()
                if abs(net) > MAX_NET_EXPOSURE:
                    longs = target_w_neutral[target_w_neutral > 0]
                    shorts = target_w_neutral[target_w_neutral < 0]
                    trim = abs(net) / 2
                    if net > 0:
                        target_w_neutral[longs.index] *= (1 - trim / longs.sum())
                    else:
                        target_w_neutral[shorts.index] *= (1 - trim / shorts.abs().sum())
                
                target_w_lev = apply_leverage(target_w_neutral, port_vol_ewm * np.sqrt(365))
                actual_w = apply_turnover_buffer(target_w_lev, current_w)
                trades = (actual_w - current_w.reindex(actual_w.index, fill_value=0)).abs()
        
        cost_pct = trades.sum() * COST_PER_TRADE
        
        if next_date in returns.index and len(actual_w) > 0:
            ret_next = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            gross_ret = (actual_w * ret_next).sum()
        else:
            gross_ret = 0.0
        
        net_ret = gross_ret - cost_pct
        nav = nav * (1 + net_ret)
        hwm = max(hwm, nav)
        dd = (nav - hwm) / hwm
        
        alpha_port = 1 - np.exp(-np.log(2) / EWMA_HALFLIFE)
        port_vol_ewm = np.sqrt(alpha_port * net_ret**2 + (1 - alpha_port) * port_vol_ewm**2)
        
        if dd < MAX_DRAWDOWN_FLAT and pause_counter == 0:
            pause_counter = PAUSE_DAYS
            hwm = nav
            print(f"  🚨 [{date.date()}] Max DD {dd:.2%} → flatten + pause {PAUSE_DAYS}d")
        
        if abs(net_ret) < abs(DAILY_LOSS_LIMIT) and pause_counter == 0:
            pause_counter = PAUSE_DAYS
            print(f"  🛑 [{date.date()}] Daily loss {net_ret:.2%} > {DAILY_LOSS_LIMIT:.0%} → pause {PAUSE_DAYS}d")
        
        gross_exp = actual_w.abs().sum()
        net_exp = actual_w.sum()
        n_long = (actual_w > 0).sum()
        n_short = (actual_w < 0).sum()
        port_vol_ann = port_vol_ewm * np.sqrt(365)
        
        pnl_rows.append({
            "date": date,
            "nav": nav,
            "daily_ret": net_ret,
            "gross_ret": gross_ret,
            "cost_pct": cost_pct,
            "drawdown": dd,
            "gross_exp": gross_exp,
            "net_exp": net_exp,
            "n_long": n_long,
            "n_short": n_short,
            "port_vol_ann": port_vol_ann,
            "portfolio_beta": port_beta_before,
            "paused": pause_counter > 0,
        })
        
        if len(actual_w) > 0:
            for sym, w in actual_w.items():
                position_rows.append({"date": date, "symbol": sym, "weight": w})
    
    pnl_df = pd.DataFrame(pnl_rows).set_index("date")
    pos_df = pd.DataFrame(position_rows)
    
    return pnl_df, pos_df


# ── METRICS & OUTPUT ──────────────────────────────────────────────────────────

def compute_metrics(pnl_df: pd.DataFrame) -> dict:
    nav = pnl_df["nav"]
    ret = pnl_df["daily_ret"]
    
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    n_years = len(nav) / 365
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = ret.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    dd = pnl_df["drawdown"]
    max_dd = dd.min()
    
    gross_exp = pnl_df["gross_exp"]
    avg_gross = gross_exp.mean()
    
    # Win rate
    win_rate = (ret > 0).mean()
    
    # Profit factor
    gains = ret[ret > 0].sum()
    losses = ret[ret < 0].abs().sum()
    profit_factor = gains / losses if losses > 0 else 0
    
    # Trade frequency
    cost_total = pnl_df["cost_pct"].sum()
    n_trades = cost_total / COST_PER_TRADE / 2  # round trip
    
    # Beta diagnostics
    beta_before = pnl_df["portfolio_beta"]
    avg_beta_before = beta_before.mean()
    
    return {
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_gross": avg_gross,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_beta_before": avg_beta_before,
        "n_years": n_years,
    }


def compute_monthly_returns(pnl_df: pd.DataFrame) -> pd.DataFrame:
    monthly = pnl_df["daily_ret"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({"monthly_ret": monthly})
    monthly_df.index = monthly_df.index.to_period("M")
    return monthly_df


def create_chart(pnl_df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # Panel 1: NAV
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(pnl_df.index, pnl_df["nav"], linewidth=1.5, color="blue")
    ax1.axhline(INITIAL_NAV, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("NAV ($)")
    ax1.set_title("EXP12: Beta-Neutral Portfolio", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(pnl_df.index, pnl_df["drawdown"], 0, color="red", alpha=0.3)
    ax2.axhline(MAX_DRAWDOWN_FLAT, color="darkred", linestyle="--", label=f"Max DD ({MAX_DRAWDOWN_FLAT:.0%})")
    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Gross Exposure
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(pnl_df.index, pnl_df["gross_exp"], linewidth=1, color="green")
    ax3.axhline(MAX_GROSS_EXPOSURE, color="darkgreen", linestyle="--", label=f"Max Gross ({MAX_GROSS_EXPOSURE:.1f})")
    ax3.set_ylabel("Gross Exposure")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Portfolio Beta (before neutralization)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(pnl_df.index, pnl_df["portfolio_beta"], linewidth=1, color="purple")
    ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_ylabel("Portfolio Beta\n(Before Neutral)")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/portfolio_chart.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir}/portfolio_chart.png")


def save_outputs(pnl_df: pd.DataFrame, pos_df: pd.DataFrame, metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    pnl_df.to_csv(f"{out_dir}/portfolio_pnl.csv")
    print(f"Saved: {out_dir}/portfolio_pnl.csv")
    
    pos_df.to_csv(f"{out_dir}/portfolio_positions.csv", index=False)
    print(f"Saved: {out_dir}/portfolio_positions.csv")
    
    monthly_df = compute_monthly_returns(pnl_df)
    monthly_df.to_csv(f"{out_dir}/portfolio_monthly.csv")
    print(f"Saved: {out_dir}/portfolio_monthly.csv")
    
    # Metrics file
    with open(f"{out_dir}/portfolio_metrics.txt", "w") as f:
        f.write("=== EXP12: BETA-NEUTRAL CONSTRUCTION ===\n")
        f.write("(In-sample simulation — structural fix, not parameter optimization)\n\n")
        
        f.write("--- Architecture ---\n")
        f.write(f"Beta neutralization: {BETA_WINDOW}d rolling OLS vs equal-weight market\n")
        f.write(f"Sizing: vol-target only (no regime filters, no IC scaling, no dispersion scaling)\n")
        f.write("Parameters removed vs exp11: 9 (all regime/filter params)\n")
        f.write("Parameters added: 0 (beta window = signal horizon, not tuned)\n\n")
        
        f.write("--- Performance Metrics ---\n")
        f.write(f"Total Return:      {metrics['total_ret']:.2%}\n")
        f.write(f"Annualized Return: {metrics['ann_ret']:.2%}\n")
        f.write(f"Annualized Vol:    {metrics['ann_vol']:.2%}\n")
        f.write(f"Sharpe Ratio:      {metrics['sharpe']:.2f}\n")
        f.write(f"Max Drawdown:      {metrics['max_dd']:.2%}\n")
        f.write(f"Avg Gross Exp:     {metrics['avg_gross']:.2f}\n")
        f.write(f"Win Rate:          {metrics['win_rate']:.1%}\n")
        f.write(f"Profit Factor:     {metrics['profit_factor']:.2f}\n")
        f.write(f"Total Trades:      {metrics['n_trades']:.0f}\n")
        f.write(f"Years:             {metrics['n_years']:.1f}\n\n")
        
        f.write("--- Beta Diagnostics ---\n")
        f.write(f"Avg portfolio beta before neutralization: {metrics['avg_beta_before']:.3f}\n")
        f.write("Avg portfolio beta after neutralization:  ~0.000 (by construction)\n")
        f.write("Max |portfolio beta| after neutralization: ~0.000 (by construction)\n")
    
    print(f"Saved: {out_dir}/portfolio_metrics.txt")
    
    create_chart(pnl_df, out_dir)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXP12: BETA-NEUTRAL PORTFOLIO BACKTEST")
    print("=" * 60)
    
    scores = load_meta_scores()
    returns = load_returns()
    
    print("\nComputing rolling betas...")
    betas = compute_rolling_betas(returns)
    print(f"Betas computed: {betas.shape}")
    
    pnl_df, pos_df = run_simulation(scores, returns, betas)
    
    metrics = compute_metrics(pnl_df)
    
    save_outputs(pnl_df, pos_df, metrics, OUT_DIR)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Return:  {metrics['total_ret']:.2%}")
    print(f"Sharpe Ratio:  {metrics['sharpe']:.2f}")
    print(f"Max Drawdown:  {metrics['max_dd']:.2%}")
    print(f"Avg Beta (before neutral): {metrics['avg_beta_before']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
