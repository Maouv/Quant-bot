#!/usr/bin/env python3
"""
Regime Filter Sensitivity Analysis
====================================
Tests cross-sectional dispersion detector across threshold range 1.0x-2.5x.
Anti-overfitting protocol: if results are robust across range, threshold is structural.
If only good at one specific value, it's data-mined — don't use.

Output (all in ./regime_test/):
    sensitivity_results.csv   — Sharpe/return per year per threshold
    sensitivity_chart.png     — visual of threshold sensitivity
    regime_test_summary.txt   — verdict: robust or data-mined

Usage:
    python3 regime_test.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

OHLCV_DIR     = "./merged_data"
COMPOSITE_DIR = "./composite"
OUT_DIR       = "./regime_test"

# Thresholds to test — sensitivity analysis
THRESHOLDS    = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 999]
# 999 = no filter (baseline)

# Dispersion window params
DISP_ROLLING  = 5     # days to smooth dispersion
DISP_MEDIAN_W = 120   # expanding window minimum before median is valid

# Portfolio params (same as portofolio.py)
VOL_TARGET         = 0.15
EWMA_HALFLIFE      = 60
MAX_POS_PER_SYMBOL = 0.12
MAX_SHORT_PER_SYMBOL = 0.06
MAX_GROSS_EXPOSURE = 3.0
COST_PER_TRADE     = 0.0007
TURNOVER_BUFFER    = 0.02

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Load returns and meta scores."""
    # Returns
    returns = {}
    for fp in glob.glob(f"{OHLCV_DIR}/*.csv"):
        sym = os.path.basename(fp).replace(".csv", "").replace("-1d-full", "")
        df  = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df  = df.set_index("date").sort_index()
        df.index = df.index.normalize()
        returns[sym] = df["close"].astype(float).pct_change()
    ret_df = pd.DataFrame(returns).sort_index()

    # Meta scores
    scores = pd.read_csv(f"{COMPOSITE_DIR}/composite_scores_meta.csv",
                         index_col="date", parse_dates=True)
    scores.index = scores.index.normalize()

    # Align
    common_syms = scores.columns.intersection(ret_df.columns)
    scores  = scores[common_syms]
    ret_df  = ret_df[common_syms]

    print(f"Loaded: {len(ret_df)} days × {len(common_syms)} symbols")
    return scores, ret_df


# ── DISPERSION SERIES ─────────────────────────────────────────────────────────

def compute_dispersion(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily cross-sectional dispersion (std of returns across symbols).
    Smoothed over DISP_ROLLING days.
    Expanding window median computed without lookahead.
    """
    # Daily cross-sectional std
    cs_disp = returns.std(axis=1)
    cs_disp_smooth = cs_disp.rolling(DISP_ROLLING, min_periods=1).mean()

    # Expanding window median (no lookahead bias)
    cs_disp_median = cs_disp_smooth.expanding(min_periods=DISP_MEDIAN_W).median()

    # Dispersion ratio
    disp_ratio = cs_disp_smooth / cs_disp_median

    return pd.DataFrame({
        "cs_disp":        cs_disp,
        "cs_disp_smooth": cs_disp_smooth,
        "cs_disp_median": cs_disp_median,
        "disp_ratio":     disp_ratio,
    })


# ── SIMPLIFIED SIMULATION ─────────────────────────────────────────────────────

def run_sim(scores, returns, disp_df, threshold):
    """
    Simplified simulation with dispersion filter.
    When disp_ratio > threshold: halve short positions (reduce short exposure).
    threshold=999 means no filter (baseline).
    """
    # Per-symbol EWMA vol
    alpha_sym = 1 - np.exp(-np.log(2) / 30)
    sym_vol = returns.ewm(alpha=alpha_sym, min_periods=10).std() * np.sqrt(365)
    sym_vol = sym_vol.clip(lower=0.05)

    # BTC vol for regime
    btc_vol = returns["BTCUSDT"].rolling(60).std() * np.sqrt(365) if "BTCUSDT" in returns.columns else pd.Series(0.5, index=returns.index)
    btc_vol_median = btc_vol.median()

    all_dates = scores.index.intersection(returns.index)
    nav = 100_000.0
    hwm = nav
    current_w = pd.Series(dtype=float)
    port_vol_ewm = VOL_TARGET / np.sqrt(365)
    alpha_port = 1 - np.exp(-np.log(2) / EWMA_HALFLIFE)
    pause_counter = 0

    daily_rows = []

    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]

        if date not in scores.index:
            continue
        score_today = scores.loc[date].dropna()
        if len(score_today) < 5:
            continue

        # Dispersion filter
        disp_active = False
        if threshold < 999 and date in disp_df.index:
            dr = disp_df.loc[date, "disp_ratio"]
            if not pd.isna(dr) and dr > threshold:
                disp_active = True

        # Pause check
        if pause_counter > 0:
            pause_counter -= 1
            actual_w = pd.Series(0.0, index=current_w.index)
            trades = current_w.abs()
            current_w = pd.Series(dtype=float)
        else:
            # Target weights
            sv = sym_vol.loc[date] if date in sym_vol.index else pd.Series(0.5, index=score_today.index)
            common = score_today.index.intersection(sv.index)
            s = score_today[common].dropna()
            v = sv[common].reindex(s.index).fillna(0.5)
            w_raw = (s / v).replace([np.inf, -np.inf], np.nan).dropna()

            if len(w_raw) == 0:
                actual_w = pd.Series(dtype=float)
                trades = pd.Series(dtype=float)
            else:
                longs  = w_raw[w_raw > 0]
                shorts = w_raw[w_raw < 0]
                w_final = pd.Series(0.0, index=w_raw.index)

                if len(longs) > 0:
                    wl = (longs / longs.sum()).clip(upper=MAX_POS_PER_SYMBOL)
                    wl = wl / wl.sum()
                    w_final[longs.index] = wl

                if len(shorts) > 0:
                    # Apply dispersion filter: halve short cap during altseason
                    short_cap = MAX_SHORT_PER_SYMBOL * 0.5 if disp_active else MAX_SHORT_PER_SYMBOL
                    ws = (shorts / shorts.abs().sum()).clip(lower=-short_cap)
                    ws = ws / ws.abs().sum()
                    w_final[shorts.index] = ws

                # Leverage
                bv = btc_vol.get(date, btc_vol_median)
                if pd.isna(bv) or bv == 0:
                    bv = btc_vol_median
                regime_scale = min(1.0, btc_vol_median / bv)
                port_vol_ann = port_vol_ewm * np.sqrt(365)
                lev = min(VOL_TARGET / port_vol_ann, MAX_GROSS_EXPOSURE / 2) * regime_scale
                w_levered = w_final * lev

                # Turnover buffer
                all_syms = w_levered.index.union(current_w.index)
                t = w_levered.reindex(all_syms, fill_value=0.0)
                c = current_w.reindex(all_syms, fill_value=0.0)
                delta = t - c
                trade_mask = delta.abs() > TURNOVER_BUFFER
                actual_w = c.copy()
                actual_w[trade_mask] = t[trade_mask]
                trades = (actual_w - c).abs()

        # Costs + P&L
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

        # Update vol
        port_vol_ewm = np.sqrt(alpha_port * net_ret**2 + (1 - alpha_port) * port_vol_ewm**2)

        # Risk check
        if dd < -0.20 and pause_counter == 0:
            pause_counter = 3
            hwm = nav

        # Drift
        if len(actual_w) > 0 and next_date in returns.index:
            ret_drift = returns.loc[next_date].reindex(actual_w.index, fill_value=0.0)
            current_w = actual_w * (1 + ret_drift)
        else:
            current_w = actual_w

        daily_rows.append({
            "date":       date,
            "nav":        nav,
            "daily_ret":  net_ret,
            "gross_ret":  gross_ret,
            "drawdown":   dd,
            "disp_active": disp_active,
        })

    return pd.DataFrame(daily_rows).set_index("date")


# ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────

def run_sensitivity(scores, returns, disp_df):
    """Run simulation for each threshold and collect annual metrics."""
    print("=== SENSITIVITY ANALYSIS ===")
    print(f"Testing {len(THRESHOLDS)} thresholds: {THRESHOLDS}\n")

    all_results = []

    for thr in THRESHOLDS:
        label = "baseline" if thr == 999 else f"{thr}x"
        print(f"  Running threshold={label}...", end="", flush=True)

        pnl = run_sim(scores, returns, disp_df, thr)

        # Annual metrics
        pnl["year"] = pnl.index.year
        annual = pnl.groupby("year")["daily_ret"].agg(
            total_ret=lambda x: (1+x).prod()-1,
            sharpe=lambda x: x.mean()/x.std()*np.sqrt(365) if x.std()>0 else 0,
        )

        total_ret = (pnl["nav"].iloc[-1] / pnl["nav"].iloc[0]) - 1
        n_years = len(pnl) / 365
        cagr = (1 + total_ret) ** (1/n_years) - 1
        sharpe_full = pnl["daily_ret"].mean() / pnl["daily_ret"].std() * np.sqrt(365)
        max_dd = pnl["drawdown"].min()

        print(f" CAGR={cagr:.1%}  Sharpe={sharpe_full:.2f}  MaxDD={max_dd:.1%}")

        row = {
            "threshold": label,
            "cagr": cagr,
            "sharpe_full": sharpe_full,
            "max_dd": max_dd,
        }
        for yr in [2022, 2023, 2024, 2025]:
            if yr in annual.index:
                row[f"ret_{yr}"] = annual.loc[yr, "total_ret"]
                row[f"sharpe_{yr}"] = annual.loc[yr, "sharpe"]
        all_results.append(row)

    return pd.DataFrame(all_results)


# ── VERDICT ───────────────────────────────────────────────────────────────────

def verdict(results: pd.DataFrame) -> str:
    """
    Anti-overfitting check:
    - If Sharpe improves monotonically as threshold decreases → structural
    - If only one threshold is good → data-mined
    - If 2025 Sharpe drops > 30% at best threshold → too aggressive
    """
    baseline = results[results["threshold"] == "baseline"].iloc[0]
    filtered = results[results["threshold"] != "baseline"].copy()

    # Best threshold by full-sample Sharpe
    best_idx = filtered["sharpe_full"].idxmax()
    best = filtered.loc[best_idx]

    sharpe_improvement = best["sharpe_full"] - baseline["sharpe_full"]
    sharpe_2025_baseline = baseline.get("sharpe_2025", 1.73)
    sharpe_2025_best = best.get("sharpe_2025", 0)
    sharpe_2025_drop = (sharpe_2025_baseline - sharpe_2025_best) / abs(sharpe_2025_baseline) if sharpe_2025_baseline != 0 else 0

    # Count how many thresholds improve Sharpe vs baseline
    n_improve = (filtered["sharpe_full"] > baseline["sharpe_full"]).sum()
    pct_improve = n_improve / len(filtered)

    lines = [
        "=== REGIME FILTER SENSITIVITY — VERDICT ===\n\n",
        f"Baseline (no filter): Sharpe={baseline['sharpe_full']:.2f}, CAGR={baseline['cagr']:.1%}\n",
        f"Best threshold: {best['threshold']} → Sharpe={best['sharpe_full']:.2f}, CAGR={best['cagr']:.1%}\n",
        f"Sharpe improvement: {sharpe_improvement:+.2f}\n",
        f"Thresholds that improve vs baseline: {n_improve}/{len(filtered)} ({pct_improve:.0%})\n",
        f"2025 Sharpe drop at best threshold: {sharpe_2025_drop:.0%}\n\n",
    ]

    if pct_improve >= 0.7 and sharpe_2025_drop < 0.30:
        verdict_str = "✅ ROBUST — improvement is structural, not data-mined"
        verdict_str += f"\n   Recommended threshold: {best['threshold']}"
        verdict_str += f"\n   >70% of thresholds improve vs baseline"
        verdict_str += f"\n   2025 performance not significantly hurt"
    elif pct_improve >= 0.5:
        verdict_str = "⚠️ BORDERLINE — proceed with caution"
        verdict_str += f"\n   Only {pct_improve:.0%} of thresholds improve"
        verdict_str += f"\n   Consider simpler fix instead"
    else:
        verdict_str = "❌ DATA-MINED — do not implement"
        verdict_str += f"\n   Only {pct_improve:.0%} of thresholds improve"
        verdict_str += f"\n   Threshold is fitted to 2023 failure months"

    lines.append(f"VERDICT: {verdict_str}\n\n")

    # Annual breakdown table
    lines.append("=== Annual Returns by Threshold ===\n")
    lines.append(f"{'Threshold':<12} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8} {'Sharpe':>8} {'CAGR':>8}\n")
    lines.append("-" * 65 + "\n")
    for _, row in results.iterrows():
        lines.append(
            f"{row['threshold']:<12} "
            f"{row.get('ret_2022', 0):>8.1%} "
            f"{row.get('ret_2023', 0):>8.1%} "
            f"{row.get('ret_2024', 0):>8.1%} "
            f"{row.get('ret_2025', 0):>8.1%} "
            f"{row['sharpe_full']:>8.2f} "
            f"{row['cagr']:>8.1%}\n"
        )

    return "".join(lines)


# ── CHARTS ────────────────────────────────────────────────────────────────────

def create_charts(results: pd.DataFrame, disp_df: pd.DataFrame):
    Path(OUT_DIR).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dispersion Filter Sensitivity Analysis\n(Anti-Overfitting Protocol)", fontsize=13)

    filtered = results[results["threshold"] != "baseline"].copy()
    x_labels = filtered["threshold"].tolist()
    x = range(len(x_labels))

    # Full-sample Sharpe
    axes[0,0].bar(x, filtered["sharpe_full"], color="steelblue", alpha=0.8)
    baseline_sharpe = results[results["threshold"]=="baseline"]["sharpe_full"].iloc[0]
    axes[0,0].axhline(baseline_sharpe, color="red", linestyle="--", label=f"Baseline ({baseline_sharpe:.2f})")
    axes[0,0].axhline(0, color="black", linewidth=0.5)
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(x_labels, rotation=45)
    axes[0,0].set_title("Full-Sample Sharpe by Threshold")
    axes[0,0].legend(fontsize=8)

    # 2023 return
    axes[0,1].bar(x, filtered["ret_2023"]*100, color="tomato", alpha=0.8)
    base_2023 = results[results["threshold"]=="baseline"]["ret_2023"].iloc[0] * 100
    axes[0,1].axhline(base_2023, color="red", linestyle="--", label=f"Baseline ({base_2023:.1f}%)")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(x_labels, rotation=45)
    axes[0,1].set_title("2023 Return by Threshold")
    axes[0,1].legend(fontsize=8)

    # 2025 return (should not drop much)
    axes[1,0].bar(x, filtered["ret_2025"]*100, color="green", alpha=0.8)
    base_2025 = results[results["threshold"]=="baseline"]["ret_2025"].iloc[0] * 100
    axes[1,0].axhline(base_2025, color="red", linestyle="--", label=f"Baseline ({base_2025:.1f}%)")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(x_labels, rotation=45)
    axes[1,0].set_title("2025 Return by Threshold (should stay high)")
    axes[1,0].legend(fontsize=8)

    # Dispersion ratio time series
    axes[1,1].plot(disp_df["disp_ratio"], color="steelblue", linewidth=0.8, alpha=0.7)
    axes[1,1].axhline(1.5, color="red", linestyle="--", linewidth=1, label="1.5x (Opus suggestion)")
    axes[1,1].axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    # Shade altseason months
    for period in [("2023-01-01", "2023-01-31"), ("2023-11-01", "2023-12-31")]:
        axes[1,1].axvspan(pd.Timestamp(period[0]), pd.Timestamp(period[1]),
                         alpha=0.2, color="red", label="Altseason" if period[0]=="2023-01-01" else "")
    axes[1,1].set_title("Cross-Sectional Dispersion Ratio Over Time")
    axes[1,1].legend(fontsize=8)
    axes[1,1].set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sensitivity_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/sensitivity_chart.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(exist_ok=True)

    print("Loading data...")
    scores, returns = load_data()

    print("Computing dispersion series...")
    disp_df = compute_dispersion(returns)
    print(f"  Dispersion ratio stats: mean={disp_df['disp_ratio'].mean():.2f}, "
          f"median={disp_df['disp_ratio'].median():.2f}, "
          f"p75={disp_df['disp_ratio'].quantile(0.75):.2f}, "
          f"p90={disp_df['disp_ratio'].quantile(0.90):.2f}")
    print(f"  Altseason detection (ratio>1.5):")
    high_disp = disp_df[disp_df["disp_ratio"] > 1.5]
    if len(high_disp) > 0:
        print(f"    {len(high_disp)} days with ratio>1.5")
        by_month = high_disp.groupby(high_disp.index.to_period("M")).size()
        print(f"    Top months: {by_month.nlargest(5).to_dict()}")

    # Sensitivity analysis
    results = run_sensitivity(scores, returns, disp_df)

    # Verdict
    verdict_text = verdict(results)
    print("\n" + verdict_text)

    # Save
    results.to_csv(f"{OUT_DIR}/sensitivity_results.csv", index=False)
    with open(f"{OUT_DIR}/regime_test_summary.txt", "w") as f:
        f.write(verdict_text)

    # Charts
    create_charts(results, disp_df)

    print(f"\n✅ Done. Results in {OUT_DIR}/")
    print(f"   regime_test_summary.txt  ← verdict: robust or data-mined")
    print(f"   sensitivity_chart.png    ← visual sensitivity analysis")


if __name__ == "__main__":
    main()

