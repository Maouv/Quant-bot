#!/usr/bin/env python3
"""
Binance Vision Metrics Downloader — Phase 1 Validation Sample
============================================================
Downloads OI + L/S ratio data for 10 liquid symbols × 3 months (2024-01 to 2024-03).
Aggregates 5-min files to daily, computes IC, prints decision gate.

Usage:
    python3 download_metrics.py              # Phase 1: validation sample
    python3 download_metrics.py --full       # Phase 2: full download (26 symbols, all years)

Output:
    merged_metrics/{SYMBOL}-metrics-daily.csv
    metrics_ic_results.csv
    metrics_ic_summary.txt
"""

import os
import sys
import glob
import time
import zipfile
import argparse
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from scipy.stats import spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_URL    = "https://data.binance.vision/data/futures/um/daily/metrics"
RAW_DIR     = "./metrics_raw"       # temp: zip files (deleted after extract)
OUT_DIR     = "./merged_metrics"    # final: daily CSVs per symbol
OHLCV_DIR   = "./merged_data"       # existing OHLCV data

RATE_SLEEP  = 0.5    # seconds between requests — be polite to Binance
MIN_ROWS    = 50     # min rows in a 5-min file to count as valid (288 = full day)
MIN_SYMBOLS = 5      # min symbols with data per date to compute IC
IC_HORIZON  = 1      # forward return horizon for validation

# Phase 1: validation sample
SYMBOLS_SAMPLE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT",
]
DATE_START_SAMPLE = date(2024, 1, 1)
DATE_END_SAMPLE   = date(2024, 3, 31)

# Phase 2: full download
SYMBOLS_FULL = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT",   "INJUSDT", "FETUSDT",
    "STXUSDT", "TIAUSDT", "SEIUSDT",  "WLDUSDT", "RNDRUSDT",
    "ATOMUSDT", "LTCUSDT", "MKRUSDT", "UNIUSDT", "AAVEUSDT",
    "SUIUSDT",
]
DATE_START_FULL = date(2022, 1, 1)
DATE_END_FULL   = date(2025, 12, 31)

# Decision gate thresholds
IC_THRESHOLD_OI = 0.02
IC_THRESHOLD_LS = 0.02

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download_one(symbol: str, date_str: str) -> str:
    """Download one daily metrics zip. Returns 'ok', 'skip', or 'miss'."""
    fname    = f"{symbol}-metrics-{date_str}.zip"
    url      = f"{BASE_URL}/{symbol}/{fname}"
    out_path = Path(RAW_DIR) / symbol / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return "skip"

    try:
        urllib.request.urlretrieve(url, out_path)
        return "ok"
    except Exception:
        return "miss"
    finally:
        time.sleep(RATE_SLEEP)


def download_all(symbols, date_start, date_end):
    """Download all symbols × date range. Shows progress per day."""
    total_ok   = 0
    total_miss = 0

    d = date_start
    while d <= date_end:
        date_str = d.strftime("%Y-%m-%d")
        day_ok = 0
        for sym in symbols:
            result = download_one(sym, date_str)
            if result == "ok":
                total_ok += 1
                day_ok   += 1
            elif result == "miss":
                total_miss += 1

        print(f"[{date_str}] downloaded={day_ok}/{len(symbols)}  "
              f"total ok={total_ok}  miss={total_miss}", flush=True)
        d += timedelta(days=1)

    print(f"\n✅ Download done: {total_ok} files, {total_miss} missing (normal)\n")

# ── AGGREGATE 5-MIN → DAILY ───────────────────────────────────────────────────

def aggregate_day(df5: pd.DataFrame) -> dict:
    """Collapse one day's 5-min data into a single daily row."""
    return {
        # OI: stock variable — use end-of-day snapshot
        "oi_value":           df5["sum_open_interest_value"].iloc[-1],
        "oi_change_pct":      (df5["sum_open_interest_value"].iloc[-1] /
                               df5["sum_open_interest_value"].iloc[0]) - 1,
        # L/S ratios: flow variable — use daily mean
        "ls_ratio_mean":      df5["count_long_short_ratio"].mean(),
        "top_trader_ls_mean": df5["sum_toptrader_long_short_ratio"].mean(),
    }


def process_symbol(symbol: str) -> pd.DataFrame | None:
    """Aggregate all daily zip files for one symbol into a DataFrame."""
    files = sorted(glob.glob(f"{RAW_DIR}/{symbol}/{symbol}-metrics-*.zip"))
    if not files:
        print(f"  ⚠️  {symbol}: no files found")
        return None

    rows = []
    for fp in files:
        try:
            with zipfile.ZipFile(fp) as z:
                df5 = pd.read_csv(z.open(z.namelist()[0]))

            if len(df5) < MIN_ROWS:
                continue  # partial day

            row = aggregate_day(df5)
            date_str = Path(fp).stem.replace(f"{symbol}-metrics-", "")
            row["date"] = pd.to_datetime(date_str)
            rows.append(row)

            # Delete zip immediately to save space
            os.remove(fp)

        except Exception as e:
            os.remove(fp) if os.path.exists(fp) else None
            continue

    if not rows:
        print(f"  ⚠️  {symbol}: no valid data after aggregation")
        return None

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Forward-fill OI only (stock variable — valid to carry forward)
    df["oi_value"]      = df["oi_value"].ffill()
    df["oi_change_pct"] = df["oi_change_pct"]  # no ffill for changes
    # L/S ratios — leave NaN (don't ffill flow variables)

    return df


def aggregate_all(symbols) -> dict:
    """Aggregate all symbols, save CSVs, return dict of DataFrames."""
    print("=== AGGREGATING DAILY DATA ===")
    Path(OUT_DIR).mkdir(exist_ok=True)
    metrics = {}

    for sym in symbols:
        df = process_symbol(sym)
        if df is not None:
            out_path = f"{OUT_DIR}/{sym}-metrics-daily.csv"
            df.to_csv(out_path)
            print(f"  ✅ {sym}: {len(df)} days → {out_path}")
            metrics[sym] = df
        # Clean up empty symbol dir
        sym_dir = Path(RAW_DIR) / sym
        if sym_dir.exists() and not any(sym_dir.iterdir()):
            sym_dir.rmdir()

    print(f"\nAggregated {len(metrics)} symbols\n")
    return metrics

# ── IC COMPUTATION ────────────────────────────────────────────────────────────

def load_ohlcv(symbols) -> dict:
    """Load existing daily OHLCV close prices for IC forward return."""
    closes = {}
    for sym in symbols:
        pattern = f"{OHLCV_DIR}/{sym}*.csv"
        files = glob.glob(pattern)
        if not files:
            continue
        df = pd.read_csv(files[0])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("date").sort_index()
        closes[sym] = df["close"].astype(float)
    return closes


def newey_west_tstat(ic_arr: np.ndarray, lags: int = 5) -> float:
    """Compute Newey-West HAC t-stat for IC series."""
    T = len(ic_arr)
    if T < lags + 2:
        return 0.0
    ic_mean    = ic_arr.mean()
    ic_demeaned = ic_arr - ic_mean
    gamma_0    = np.sum(ic_demeaned ** 2) / T
    hac_var    = gamma_0
    for lag in range(1, lags + 1):
        weight  = 1.0 - lag / (lags + 1)
        gamma_l = np.sum(ic_demeaned[lag:] * ic_demeaned[:-lag]) / T
        hac_var += 2 * weight * gamma_l
    hac_var = max(hac_var, 1e-12)
    return ic_mean / np.sqrt(hac_var / T)


def compute_metrics_ic(metrics: dict, closes: dict) -> pd.DataFrame:
    """
    Compute cross-sectional IC for 3 metrics signals vs h=1 forward return.

    Signals:
        oi_price_signal   = -(oi_pct_5d - price_pct_5d)   [contrarian divergence]
        ls_contrarian     = -ls_ratio_mean                  [fade retail crowd]
        top_trader_contr  = -top_trader_ls_mean             [fade or follow — let IC tell]
    """
    print("=== COMPUTING IC ===")

    # Build aligned panel: date × symbol
    all_dates = sorted(set.union(*[set(df.index) for df in metrics.values()]))

    # Forward returns panel (h=1)
    fwd_ret = {}
    for sym, close_s in closes.items():
        fwd_ret[sym] = close_s.pct_change(IC_HORIZON).shift(-IC_HORIZON)
    fwd_df = pd.DataFrame(fwd_ret, index=pd.DatetimeIndex(all_dates)).sort_index()

    # Winsorize forward returns per date
    def winsor(x):
        if x.isna().all(): return x
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        return x.clip(lo, hi)
    fwd_df = fwd_df.apply(winsor, axis=1)

    # Build signal panels
    signals_def = {
        "oi_price_signal":  {},
        "ls_contrarian":    {},
        "top_trader_contr": {},
    }

    for sym, mdf in metrics.items():
        if sym not in closes:
            continue
        close_s = closes[sym]

        # OI-price divergence: -(OI_pct_5d - price_pct_5d)
        oi_pct_5d    = mdf["oi_value"].pct_change(5)
        price_pct_5d = close_s.reindex(mdf.index).pct_change(5)
        signals_def["oi_price_signal"][sym]  = -(oi_pct_5d - price_pct_5d)

        # L/S contrarian: fade retail crowd
        signals_def["ls_contrarian"][sym]    = -mdf["ls_ratio_mean"]

        # Top trader L/S: test direction (contrarian here, IC sign will reveal truth)
        signals_def["top_trader_contr"][sym] = -mdf["top_trader_ls_mean"]

    results = []
    for sig_name, sig_dict in signals_def.items():
        sig_df = pd.DataFrame(sig_dict)

        # Cross-sectional z-score per day
        def zscore(x):
            if x.std() == 0 or x.isna().all(): return x * 0
            return ((x - x.mean()) / x.std()).clip(-3, 3)
        sig_df = sig_df.apply(zscore, axis=1)

        ics = []
        for dt in sig_df.index:
            if dt not in fwd_df.index:
                continue
            sig_row = sig_df.loc[dt].dropna()
            fwd_row = fwd_df.loc[dt].dropna()
            common  = sig_row.index.intersection(fwd_row.index)
            if len(common) < MIN_SYMBOLS:
                continue
            x = sig_row[common].values
            y = fwd_row[common].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            ic, _ = spearmanr(x, y)
            ics.append(ic)

        if not ics:
            continue

        ic_arr  = np.array(ics)
        ic_mean = ic_arr.mean()
        ic_med  = np.median(ic_arr)
        ic_std  = ic_arr.std()
        icir    = ic_mean / ic_std if ic_std > 0 else 0
        t_nw    = newey_west_tstat(ic_arr)
        t_std   = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
        pct_pos = (ic_arr > 0).mean()

        print(f"  {sig_name:<22} IC={ic_mean:+.4f}  median={ic_med:+.4f}  "
              f"NW_t={t_nw:+.2f}  std_t={t_std:+.2f}  "
              f"ICIR={icir:.3f}  %pos={pct_pos:.0%}  n={len(ics)}")

        results.append({
            "signal":    sig_name,
            "IC_mean":   ic_mean,
            "IC_median": ic_med,
            "IC_std":    ic_std,
            "ICIR":      icir,
            "t_stat_NW": t_nw,
            "t_stat_std":t_std,
            "pct_pos":   pct_pos,
            "n_days":    len(ics),
        })

    return pd.DataFrame(results)

# ── DECISION GATE ─────────────────────────────────────────────────────────────

def decision_gate(ic_df: pd.DataFrame):
    """Print go/no-go verdict and save summary."""
    print("\n=== DECISION GATE ===")

    def check(signal_name, threshold):
        row = ic_df[ic_df["signal"] == signal_name]
        if row.empty:
            return None, False
        ic  = row["IC_mean"].iloc[0]
        t   = row["t_stat_NW"].iloc[0]
        ok  = abs(ic) > threshold and abs(t) > 1.5
        tag = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {signal_name:<22} IC={ic:+.4f}  NW_t={t:+.2f}  threshold={threshold}  → {tag}")
        return ic, ok

    ic_oi, ok_oi = check("oi_price_signal",  IC_THRESHOLD_OI)
    ic_ls, ok_ls = check("ls_contrarian",    IC_THRESHOLD_LS)
    _,     ok_tt = check("top_trader_contr", IC_THRESHOLD_LS)

    print()
    if ok_oi and ok_ls:
        verdict = "✅ PROCEED — both OI and L/S pass. Run full download."
        cmd     = "python3 download_metrics.py --full"
    elif ok_oi or ok_ls:
        verdict = "⚠️  PARTIAL — at least one signal passes. Full download still recommended."
        cmd     = "python3 download_metrics.py --full"
    else:
        verdict = "❌ SKIP — neither OI nor L/S passes threshold. Focus on composite.py."
        cmd     = "python3 composite.py"
    
    print(f"  VERDICT: {verdict}")
    print(f"  NEXT:    {cmd}\n")

    # Save summary
    lines = [
        "=== Binance Vision Metrics — Phase 1 Validation ===\n",
        f"Symbols:  {', '.join(SYMBOLS_SAMPLE)}\n",
        f"Period:   {DATE_START_SAMPLE} to {DATE_END_SAMPLE}\n\n",
        "=== IC Results (h=1) ===\n",
        f"{'Signal':<22} {'IC_mean':>8} {'IC_median':>10} {'NW_t':>7} {'std_t':>7} {'ICIR':>7} {'%pos':>6} {'n':>5}\n",
        "-" * 75 + "\n",
    ]
    for _, row in ic_df.iterrows():
        lines.append(
            f"{row['signal']:<22} {row['IC_mean']:>8.4f} {row['IC_median']:>10.4f} "
            f"{row['t_stat_NW']:>7.2f} {row['t_stat_std']:>7.2f} {row['ICIR']:>7.3f} "
            f"{row['pct_pos']:>6.0%} {int(row['n_days']):>5}\n"
        )
    lines += ["\n", f"VERDICT: {verdict}\n", f"NEXT:    {cmd}\n"]

    with open("metrics_ic_summary.txt", "w") as f:
        f.writelines(lines)

    ic_df.to_csv("metrics_ic_results.csv", index=False)
    print("  Saved: metrics_ic_summary.txt, metrics_ic_results.csv")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Phase 2: full download (26 symbols, 2022-2025)")
    args = parser.parse_args()

    if args.full:
        symbols    = SYMBOLS_FULL
        date_start = DATE_START_FULL
        date_end   = DATE_END_FULL
        print(f"=== PHASE 2: FULL DOWNLOAD ({len(symbols)} symbols, {date_start} to {date_end}) ===\n")
    else:
        symbols    = SYMBOLS_SAMPLE
        date_start = DATE_START_SAMPLE
        date_end   = DATE_END_SAMPLE
        print(f"=== PHASE 1: VALIDATION SAMPLE ({len(symbols)} symbols, {date_start} to {date_end}) ===\n")

    Path(RAW_DIR).mkdir(exist_ok=True)

    # 1. Download
    print("=== DOWNLOADING ===")
    download_all(symbols, date_start, date_end)

    # 2. Aggregate 5-min → daily
    metrics = aggregate_all(symbols)

    # Phase 2 stops here — no IC needed, just merge data
    if args.full:
        print("✅ Full download complete. Merged CSVs in ./merged_metrics/")
        print("   Next: re-run ic_research.py to include metrics signals.")
        return

    # 3. Phase 1 only: compute IC + decision gate
    closes = load_ohlcv(symbols)
    if not closes:
        print("❌ Could not load OHLCV data from ./merged_data/ — check path")
        sys.exit(1)

    ic_df = compute_metrics_ic(metrics, closes)
    if ic_df.empty:
        print("❌ No IC results — check data alignment")
        sys.exit(1)

    decision_gate(ic_df)


if __name__ == "__main__":
    main()
