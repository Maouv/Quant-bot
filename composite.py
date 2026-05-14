#!/usr/bin/env python3
"""
Composite Signal Builder
========================
Builds Fast and Slow composite signals from individual IC-weighted signals.
Uses expanding window weight estimation — no lookahead bias.

Output (all in ./composite/):
    composite_scores.csv      — daily composite score per symbol (input for portfolio.py)
    composite_ic.csv          — IC of composite vs individual signals
    composite_weights.csv     — expanding window weights over time
    composite_chart.png       — composite IC vs individual signals bar chart
    composite_decay_chart.png — IC decay: composite vs best individual

Usage:
    python3 composite.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

OHLCV_DIR    = "./merged_data"
METRICS_DIR  = "./merged_metrics"
OUT_DIR      = "./composite"

# Expanding window: min days before a signal gets non-zero weight
MIN_DAYS_WEIGHT = 120

# t-stat threshold to include signal in composite (expanding window)
TSTAT_THRESHOLD = 1.0

# Horizons to compute composite IC on
IC_DECAY_HORIZONS = [1, 2, 3, 5, 10, 20]

MIN_SYMBOLS  = 10   # min symbols per date for IC computation
CLIP_ZSCORE  = 3.0
WINSOR_PCTS  = (0.01, 0.99)

# ── SIGNAL DEFINITIONS & FIXED WEIGHTS (Opus validated) ──────────────────────
#
# Fast composite (h=1): 3 signals only — trim dead weight per Opus Plan 2
# Drop: reversal_1d (NW_t=2.64, below 2.5 threshold) and volume_compression (OOS drop)
# Keep: volatility + liquidity (structural) + reversal_1w (diversifier, NW_t=2.92)
FAST_SIGNALS = [
    "volatility",
    "liquidity",
    "reversal_1w",
]
FAST_WEIGHTS = {
    "volatility":  (+1, 0.42),  # ICIR=0.186, dominant structural factor
    "liquidity":   (-1, 0.36),  # ICIR=0.156, second structural factor
    "reversal_1w": (+1, 0.22),  # ICIR=0.074, mean-reversion diversifier
}

# Slow composite (h=20): 4 signals, ls_contrarian dropped (OOS collapse 93%)
# Weights: ICIR-proportional at h=20, conservative reallocation per Opus Q4
SLOW_SIGNALS = [
    "volatility",
    "liquidity",
    "momentum_30d",
    "taker_buy_contrarian",
]
SLOW_WEIGHTS = {
    "volatility":          (+1, 0.38),  # ICIR_h20=0.467, dominant
    "liquidity":           (-1, 0.32),  # ICIR_h20=0.453, strong
    "momentum_30d":        (+1, 0.18),  # ICIR_h20=0.282, solid OOS
    "taker_buy_contrarian":(-1, 0.12),  # ICIR_h20=0.408, keep conservative (93% OOS boost concern)
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_ohlcv() -> dict:
    """Load all OHLCV CSVs into {symbol: DataFrame}."""
    symbols = {}
    for fp in glob.glob(f"{OHLCV_DIR}/*.csv"):
        sym = os.path.basename(fp).replace(".csv", "").replace("-1d-full", "")
        df  = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df  = df.set_index("date").sort_index()
        df.index = df.index.normalize()
        for col in ["close", "high", "low", "volume", "taker_buy_volume"]:
            df[col] = df[col].astype(float)
        symbols[sym] = df
    print(f"Loaded {len(symbols)} OHLCV symbols")
    return symbols


def load_metrics() -> dict:
    """Load metrics daily CSVs into {symbol: DataFrame}. Optional."""
    metrics = {}
    for fp in glob.glob(f"{METRICS_DIR}/*-metrics-daily.csv"):
        sym = os.path.basename(fp).replace("-metrics-daily.csv", "")
        df  = pd.read_csv(fp, index_col="date", parse_dates=True)
        df.index = df.index.normalize()
        metrics[sym] = df
    if metrics:
        print(f"Loaded metrics for {len(metrics)} symbols")
    else:
        print("No metrics data found — running OHLCV signals only")
    return metrics


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

def compute_signals(symbols: dict, metrics: dict) -> dict:
    """
    Compute all signals. Returns {signal_name: DataFrame(date x symbol)}.
    Metrics signals added only if data available.
    """
    print("Computing signals...")
    sig = {s: {} for s in [
        "volatility", "liquidity", "volume_compression",
        "reversal_1w", "taker_buy_contrarian", "momentum_30d",
        "clv", "taker_imbalance_momentum", "intraday_range_ratio",
    ]}

    for sym, df in symbols.items():
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        taker  = df["taker_buy_volume"]
        hl     = (high - low).replace(0, np.nan)
        pct    = close.pct_change()

        sig["volatility"][sym]             = -pct.rolling(30).std()
        sig["liquidity"][sym]              = (volume / hl).rolling(30).mean()
        sig["volume_compression"][sym]     = volume / volume.rolling(30).mean()
        sig["reversal_1w"][sym]            = -close.pct_change(7)
        sig["taker_buy_contrarian"][sym]   = -(taker / volume).rolling(3).mean()
        sig["momentum_30d"][sym]           = close / close.shift(30) - 1
        sig["clv"][sym]                    = ((close - low) / hl).rolling(5).mean()
        sig["taker_imbalance_momentum"][sym] = (taker / volume).diff(5)
        sig["intraday_range_ratio"][sym]   = (hl / close).rolling(5).mean() / (hl / close).rolling(30).mean()

    # Metrics signals — only for symbols that have data
    if metrics:
        sig["ls_contrarian"]    = {}
        sig["top_trader_contr"] = {}
        for sym, mdf in metrics.items():
            if sym not in symbols:
                continue
            close_s = symbols[sym]["close"]
            sig["ls_contrarian"][sym]    = -mdf["ls_ratio_mean"]
            sig["top_trader_contr"][sym] = -mdf["top_trader_ls_mean"]

    # Convert to DataFrames
    for name in list(sig.keys()):
        if not sig[name]:
            del sig[name]
            continue
        sig[name] = pd.DataFrame(sig[name])

    print(f"Computed {len(sig)} signals")
    return sig


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per day, clipped to [-3, 3]."""
    def _zs(row):
        s = row.std()
        if s == 0 or row.isna().all():
            return row * 0
        return ((row - row.mean()) / s).clip(-CLIP_ZSCORE, CLIP_ZSCORE)
    return df.apply(_zs, axis=1)


def make_fwd_returns(symbols: dict, horizons: list) -> dict:
    """Build forward return panels per horizon."""
    all_dates = sorted(set.union(*[set(df.index) for df in symbols.values()]))
    fwd = {}
    for h in horizons:
        panel = {}
        for sym, df in symbols.items():
            panel[sym] = df["close"].pct_change(h).shift(-h)
        df_h = pd.DataFrame(panel, index=pd.DatetimeIndex(all_dates))

        # Winsorize per date
        def winsor(row):
            if row.isna().all(): return row
            lo, hi = row.quantile(WINSOR_PCTS[0]), row.quantile(WINSOR_PCTS[1])
            return row.clip(lo, hi)
        fwd[h] = df_h.apply(winsor, axis=1)
    return fwd


# ── EXPANDING WINDOW WEIGHTS ──────────────────────────────────────────────────

def newey_west_tstat(ic_arr: np.ndarray, lags: int = 5) -> float:
    T = len(ic_arr)
    if T < lags + 2:
        return 0.0
    mu  = ic_arr.mean()
    d   = ic_arr - mu
    g0  = np.sum(d ** 2) / T
    hac = g0
    for l in range(1, lags + 1):
        w    = 1.0 - l / (lags + 1)
        gl   = np.sum(d[l:] * d[:-l]) / T
        hac += 2 * w * gl
    hac = max(hac, 1e-12)
    return mu / np.sqrt(hac / T)


def compute_expanding_weights(ic_series: dict, all_dates: list) -> pd.DataFrame:
    """
    For each date, compute IC-weighted signal weights using all data up to that date.
    Returns DataFrame(date x signal) of weights.

    Rules:
    - Need MIN_DAYS_WEIGHT days of IC history before assigning non-zero weight
    - NW t-stat must exceed TSTAT_THRESHOLD
    - Never flip sign from overall expectation (if IC goes opposite, set to 0)
    - Weights normalized so sum(|w|) = 1
    """
    print("Computing expanding window weights...")
    signals = list(ic_series.keys())
    weight_rows = []

    # Pre-compute full-sample IC sign (prior expectation — never flip from this)
    prior_sign = {}
    for sig, ic_ts in ic_series.items():
        arr = np.array(ic_ts)
        prior_sign[sig] = np.sign(arr.mean()) if len(arr) > 0 else 0

    for i, date in enumerate(all_dates):
        row = {"date": date}
        raw_weights = {}

        for sig, ic_ts in ic_series.items():
            # Only use IC up to this date (expanding window)
            ic_up_to = [ic for d, ic in zip(all_dates, ic_ts) if d <= date]

            if len(ic_up_to) < MIN_DAYS_WEIGHT:
                raw_weights[sig] = 0.0
                continue

            arr     = np.array(ic_up_to)
            ic_mean = arr.mean()
            t       = newey_west_tstat(arr)

            # Don't include if not significant
            if abs(t) < TSTAT_THRESHOLD:
                raw_weights[sig] = 0.0
                continue

            # Don't flip sign from prior expectation
            if prior_sign[sig] != 0 and np.sign(ic_mean) != prior_sign[sig]:
                raw_weights[sig] = 0.0
                continue

            raw_weights[sig] = ic_mean

        # Normalize: sum(|w|) = 1
        total = sum(abs(v) for v in raw_weights.values())
        if total > 0:
            for sig in signals:
                row[sig] = raw_weights.get(sig, 0.0) / total
        else:
            for sig in signals:
                row[sig] = 0.0

        weight_rows.append(row)

    df_w = pd.DataFrame(weight_rows).set_index("date")
    return df_w


# ── IC COMPUTATION (single signal or composite) ───────────────────────────────

def compute_ic_series(signal_df: pd.DataFrame, fwd_df: pd.DataFrame) -> list:
    """Compute daily IC between a signal panel and forward return panel."""
    ics = []
    for date in signal_df.index:
        if date not in fwd_df.index:
            ics.append(np.nan)
            continue
        sig_row = signal_df.loc[date].dropna()
        fwd_row = fwd_df.loc[date].dropna()
        common  = sig_row.index.intersection(fwd_row.index)
        if len(common) < MIN_SYMBOLS:
            ics.append(np.nan)
            continue
        x, y = sig_row[common].values, fwd_row[common].values
        if np.std(x) == 0 or np.std(y) == 0:
            ics.append(np.nan)
            continue
        ic, _ = spearmanr(x, y)
        ics.append(ic)
    return ics


def summarize_ic(ic_arr: np.ndarray, label: str) -> dict:
    valid = ic_arr[~np.isnan(ic_arr)]
    if len(valid) == 0:
        return {}
    ic_mean = valid.mean()
    ic_med  = np.median(valid)
    ic_std  = valid.std()
    icir    = ic_mean / ic_std if ic_std > 0 else 0
    t_nw    = newey_west_tstat(valid)
    t_std   = ic_mean / (ic_std / np.sqrt(len(valid))) if ic_std > 0 else 0
    pct_pos = (valid > 0).mean()
    print(f"  {label:<28} IC={ic_mean:+.4f}  med={ic_med:+.4f}  "
          f"NW_t={t_nw:+.2f}  ICIR={icir:.3f}  %pos={pct_pos:.0%}  n={len(valid)}")
    return {
        "label": label, "IC_mean": ic_mean, "IC_median": ic_med,
        "IC_std": ic_std, "ICIR": icir, "t_stat_NW": t_nw,
        "t_stat_std": t_std, "pct_pos": pct_pos, "n_days": len(valid),
    }


# ── BUILD COMPOSITE ───────────────────────────────────────────────────────────

def build_composite(
    name: str,
    signal_names: list,
    fixed_weights: dict,
    signals: dict,
    fwd_returns: dict,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Build one composite (Fast or Slow) using fixed Opus-validated weights.

    Returns:
        scores_df   — date x symbol composite scores (z-scored)
        weights_df  — static weights as single-row DataFrame
        ic_summary  — list of IC summary dicts (composite + individuals)
    """
    print(f"\n=== BUILDING {name.upper()} COMPOSITE (h={horizon}) ===")

    # Filter to available signals only
    available = [s for s in signal_names if s in signals]
    missing   = [s for s in signal_names if s not in signals]
    if missing:
        print(f"  ⚠️  Signals not available: {missing}")
    print(f"  Using signals: {available}")

    # Renormalize weights for available signals only
    total_w = sum(fixed_weights[s][1] for s in available if s in fixed_weights)
    weights = {}
    for s in available:
        if s not in fixed_weights:
            continue
        sign, w = fixed_weights[s]
        weights[s] = (sign, w / total_w)  # renormalized
        print(f"  {s:<25} sign={'+' if sign>0 else '-'}  weight={w/total_w:.3f}")

    fwd_df    = fwd_returns[horizon]
    all_dates = sorted(set.union(*[set(df.index) for df in signals.values()
                                   if isinstance(df, pd.DataFrame)]))

    # Z-score each signal
    z_signals = {s: zscore_cs(signals[s]) for s in available}

    # Compute individual IC series
    print(f"\n  Individual signal IC (h={horizon}):")
    ic_summaries = []
    for sig in available:
        ic_ts  = compute_ic_series(z_signals[sig], fwd_df)
        ic_arr = np.array(ic_ts)
        summary = summarize_ic(ic_arr, sig)
        if summary:
            summary["composite"] = name
            summary["type"]      = "individual"
            ic_summaries.append(summary)

    # Build composite score per day using fixed weights
    print(f"\n  Building composite scores...")
    composite_rows = {}
    for date in all_dates:
        score = None
        for sig in available:
            if sig not in weights:
                continue
            if date not in z_signals[sig].index:
                continue
            sign, w = weights[sig]
            sig_row = z_signals[sig].loc[date] * sign  # apply direction
            if score is None:
                score = sig_row * w
            else:
                score = score.add(sig_row * w, fill_value=0)

        if score is None:
            continue
        # Only keep symbols with enough signal coverage (>50% signals present)
        n_signals_present = sum(
            1 for s in available
            if s in z_signals and date in z_signals[s].index
            and z_signals[s].loc[date].notna().any()
        )
        if n_signals_present < max(2, len(available) // 2):
            continue
        composite_rows[date] = score

    scores_df = pd.DataFrame(composite_rows).T
    scores_df.index.name = "date"
    scores_df = zscore_cs(scores_df)  # re-normalize cross-sectionally

    # Compute composite IC
    print(f"\n  Composite IC (h={horizon}):")
    comp_ic_ts  = compute_ic_series(scores_df, fwd_df)
    comp_ic_arr = np.array(comp_ic_ts)
    summary = summarize_ic(comp_ic_arr, f"★ {name} COMPOSITE")
    if summary:
        summary["composite"] = name
        summary["type"]      = "composite"
        ic_summaries.append(summary)

    # Weights as DataFrame for saving
    weights_df = pd.DataFrame([{
        "composite": name,
        "horizon":   horizon,
        **{s: w for s, (_, w) in weights.items()},
    }])

    return scores_df, weights_df, ic_summaries


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def create_charts(ic_df: pd.DataFrame, fast_scores: pd.DataFrame,
                  slow_scores: pd.DataFrame, fwd_returns: dict,
                  signals: dict):
    print("\nCreating charts...")
    Path(OUT_DIR).mkdir(exist_ok=True)

    # Chart 1: IC comparison — composite vs individual at h=1
    h1 = ic_df[ic_df["composite"] == "Fast"].copy()
    h1 = h1.sort_values("IC_mean", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = []
    for _, row in h1.iterrows():
        if row["type"] == "composite":
            colors.append("gold")
        elif row["IC_mean"] > 0:
            colors.append("steelblue")
        else:
            colors.append("tomato")

    y_pos = range(len(h1))
    ax.barh(y_pos, h1["IC_mean"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(h1["label"])
    ax.set_xlabel("IC Mean")
    ax.set_title("Fast Composite vs Individual Signals — IC Mean (h=1)\n★ = Composite  |  Gold = Composite")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(-0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    for i, (ic, t) in enumerate(zip(h1["IC_mean"], h1["t_stat_NW"])):
        ax.text(ic + (0.001 if ic >= 0 else -0.001), i,
                f"{ic:.4f} (NW t={t:.2f})", va="center", fontsize=8,
                ha="left" if ic >= 0 else "right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/composite_chart.png", dpi=150)
    plt.close()

    # Chart 2: IC decay — Fast composite vs top 3 individual signals
    fig, ax = plt.subplots(figsize=(11, 6))

    # Recompute IC decay for fast composite
    fwd_h = {h: fwd_returns[h] for h in [1, 2, 3, 5, 10, 20]}
    comp_decay = []
    for h, fwd_df in fwd_h.items():
        ic_ts  = compute_ic_series(fast_scores, fwd_df)
        ic_arr = np.array(ic_ts)
        valid  = ic_arr[~np.isnan(ic_arr)]
        comp_decay.append(valid.mean() if len(valid) > 0 else np.nan)

    ax.plot([1,2,3,5,10,20], comp_decay, marker="o", linewidth=2.5,
            color="gold", label="★ Fast Composite", zorder=5)

    # Top 3 individual signals by |IC| at h=1
    indiv = ic_df[(ic_df["composite"] == "Fast") & (ic_df["type"] == "individual")]
    top3  = indiv.nlargest(3, "IC_mean", keep="first")["label"].tolist()
    colors_top = ["steelblue", "tomato", "green"]

    for sig, col in zip(top3, colors_top):
        if sig not in signals:
            continue
        z_sig  = zscore_cs(signals[sig])
        decay  = []
        for h, fwd_df in fwd_h.items():
            ic_ts  = compute_ic_series(z_sig, fwd_df)
            ic_arr = np.array(ic_ts)
            valid  = ic_arr[~np.isnan(ic_arr)]
            decay.append(valid.mean() if len(valid) > 0 else np.nan)
        ax.plot([1,2,3,5,10,20], decay, marker="o", linewidth=1.5,
                color=col, alpha=0.7, label=sig)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("IC Mean")
    ax.set_title("IC Decay: Fast Composite vs Top Individual Signals")
    ax.set_xscale("log")
    ax.set_xticks([1,2,3,5,10,20])
    ax.set_xticklabels([1,2,3,5,10,20])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/composite_decay_chart.png", dpi=150)
    plt.close()

    print(f"  Saved: {OUT_DIR}/composite_chart.png")
    print(f"  Saved: {OUT_DIR}/composite_decay_chart.png")


# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────

def save_outputs(fast_scores, slow_scores, meta_scores, fast_weights, slow_weights, ic_df, meta_ic_df):
    Path(OUT_DIR).mkdir(exist_ok=True)

    # Individual composite scores
    fast_scores.to_csv(f"{OUT_DIR}/composite_scores_fast.csv")
    slow_scores.to_csv(f"{OUT_DIR}/composite_scores_slow.csv")
    meta_scores.to_csv(f"{OUT_DIR}/composite_scores_meta.csv")

    # Combined long format — portfolio.py reads this
    fast_stack = fast_scores.stack().reset_index()
    fast_stack.columns = ["date", "symbol", "score_fast"]
    slow_stack = slow_scores.stack().reset_index()
    slow_stack.columns = ["date", "symbol", "score_slow"]
    meta_stack = meta_scores.stack().reset_index()
    meta_stack.columns = ["date", "symbol", "score_meta"]

    combined = fast_stack.merge(slow_stack, on=["date", "symbol"], how="outer")
    combined = combined.merge(meta_stack, on=["date", "symbol"], how="outer")
    combined.to_csv(f"{OUT_DIR}/composite_scores.csv", index=False)

    # Weights
    fast_weights.to_csv(f"{OUT_DIR}/composite_weights_fast.csv")
    slow_weights.to_csv(f"{OUT_DIR}/composite_weights_slow.csv")

    # IC summary
    ic_df.to_csv(f"{OUT_DIR}/composite_ic.csv", index=False)
    if not meta_ic_df.empty:
        meta_ic_df.to_csv(f"{OUT_DIR}/composite_ic_meta.csv", index=False)

    # Text summary
    with open(f"{OUT_DIR}/composite_summary.txt", "w") as f:
        f.write("=== COMPOSITE SIGNAL SUMMARY ===\n\n")
        for comp_name in ["Fast", "Slow"]:
            f.write(f"--- {comp_name} Composite ---\n")
            sub = ic_df[ic_df["composite"] == comp_name].copy()
            sub = sub.sort_values("IC_mean", key=abs, ascending=False)
            f.write(f"{'Label':<30} {'IC_mean':>8} {'IC_med':>8} {'NW_t':>7} "
                    f"{'ICIR':>7} {'%pos':>6} {'type':>12}\n")
            f.write("-" * 82 + "\n")
            for _, row in sub.iterrows():
                f.write(f"{row['label']:<30} {row['IC_mean']:>8.4f} {row['IC_median']:>8.4f} "
                        f"{row['t_stat_NW']:>7.2f} {row['ICIR']:>7.3f} "
                        f"{row['pct_pos']:>6.0%} {row['type']:>12}\n")
            f.write("\n")

        if not meta_ic_df.empty:
            f.write("--- Meta-Composite (Fast 36% + Slow 64%) ---\n")
            f.write(f"{'Label':<30} {'IC_mean':>8} {'IC_med':>8} {'NW_t':>7} "
                    f"{'ICIR':>7} {'%pos':>6}\n")
            f.write("-" * 75 + "\n")
            for _, row in meta_ic_df.iterrows():
                f.write(f"{row['label']:<30} {row['IC_mean']:>8.4f} {row['IC_median']:>8.4f} "
                        f"{row['t_stat_NW']:>7.2f} {row['ICIR']:>7.3f} "
                        f"{row['pct_pos']:>6.0%}\n")

    print(f"\n✅ Outputs saved to {OUT_DIR}/")
    print(f"   composite_scores.csv         ← input for portfolio.py (fast+slow+meta)")
    print(f"   composite_scores_meta.csv    ← meta-composite scores")
    print(f"   composite_ic_meta.csv        ← meta IC across horizons")
    print(f"   composite_summary.txt")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(exist_ok=True)

    # Load data
    symbols = load_ohlcv()
    metrics = load_metrics()

    # Compute all signals
    signals = compute_signals(symbols, metrics)

    # Forward returns for all horizons
    print("Computing forward returns...")
    fwd_returns = make_fwd_returns(symbols, IC_DECAY_HORIZONS)

    # Build Fast composite (h=1)
    fast_scores, fast_weights, fast_ic = build_composite(
        name="Fast",
        signal_names=FAST_SIGNALS,
        fixed_weights=FAST_WEIGHTS,
        signals=signals,
        fwd_returns=fwd_returns,
        horizon=1,
    )

    # Build Slow composite (h=20)
    slow_scores, slow_weights, slow_ic = build_composite(
        name="Slow",
        signal_names=SLOW_SIGNALS,
        fixed_weights=SLOW_WEIGHTS,
        signals=signals,
        fwd_returns=fwd_returns,
        horizon=20,
    )

    # Combine IC results
    ic_df = pd.DataFrame(fast_ic + slow_ic)

    # Build Meta-Composite (Plan 1 — blend Fast + Slow)
    print("\n=== BUILDING META-COMPOSITE (Fast + Slow blend) ===")
    # ICIR-proportional weights: Fast=0.213, Slow=0.381
    # Fast/(Fast+Slow) = 0.213/0.594 = 0.36, Slow = 0.64
    META_WEIGHT_FAST = 0.36
    META_WEIGHT_SLOW = 0.64

    # Align on common dates and symbols
    common_dates = fast_scores.index.intersection(slow_scores.index)
    common_syms  = fast_scores.columns.intersection(slow_scores.columns)
    fast_aligned = fast_scores.loc[common_dates, common_syms]
    slow_aligned = slow_scores.loc[common_dates, common_syms]

    # Slow score smoothed — rebalances weekly not daily
    slow_smoothed = slow_aligned.rolling(5, min_periods=1).mean()

    # Blend
    meta_scores = META_WEIGHT_FAST * fast_aligned + META_WEIGHT_SLOW * slow_smoothed

    # Re-normalize cross-sectionally
    meta_scores = zscore_cs(meta_scores)

    # Compute meta-composite IC at h=1 and h=5
    print(f"  Meta-composite IC:")
    meta_ic_results = []
    for h in [1, 2, 3, 5, 10, 20]:
        ic_ts  = compute_ic_series(meta_scores, fwd_returns[h])
        ic_arr = np.array(ic_ts)
        valid  = ic_arr[~np.isnan(ic_arr)]
        if len(valid) == 0:
            continue
        summary = summarize_ic(valid, f"★ META h={h}")
        if summary:
            summary["composite"] = "Meta"
            summary["type"]      = "composite"
            meta_ic_results.append(summary)

    meta_ic_df = pd.DataFrame(meta_ic_results)

    # Correlation between Fast and Slow daily scores
    corr = fast_aligned.values.flatten()
    slow_flat = slow_aligned.values.flatten()
    mask = ~(np.isnan(corr) | np.isnan(slow_flat))
    if mask.sum() > 0:
        correlation = np.corrcoef(corr[mask], slow_flat[mask])[0, 1]
        print(f"\n  Fast-Slow score correlation: {correlation:.3f} (target < 0.5)")

    # Charts
    create_charts(ic_df, fast_scores, slow_scores, fwd_returns, signals)

    # Save all outputs
    save_outputs(fast_scores, slow_scores, meta_scores, fast_weights, slow_weights, ic_df, meta_ic_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
