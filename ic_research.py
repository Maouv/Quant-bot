#!/usr/bin/env python3
"""Cross-sectional factor IC analysis for crypto using local OHLCV data."""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
import matplotlib.pyplot as plt
from statsmodels.stats.sandwich_covariance import cov_hac

# === PARAMETERS ===
REVERSAL_SHORT = 1
REVERSAL_LONG = 7
MOMENTUM_MED = 30
MOMENTUM_LONG = 90
VOL_WINDOW = 30
VOL_SHORT = 5
LIQUIDITY_WIN = 30
VOLUME_WIN = 30
PRICE_HIGH_WIN = 30
TAKER_WIN = 3
MIN_SYMBOLS_PER_DATE = 15
WINSOR_PCTS = (0.01, 0.99)
IC_DECAY_HORIZONS = [1, 2, 3, 5, 10, 20]
IN_SAMPLE_RATIO = 0.7
CLIP_ZSCORE = 3.0

# === DATA LOADING ===
def load_data(data_dir='./merged_data/'):
    """Load all CSV files from directory into {symbol: DataFrame} dict."""
    symbols = {}
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"Loading {len(csv_files)} CSV files...")
    
    for fp in csv_files:
        symbol = os.path.basename(fp).replace('.csv', '').replace('-1d-full', '')
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('date').sort_index()
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['quote_volume'] = df['quote_volume'].astype(float)
        df['taker_buy_volume'] = df['taker_buy_volume'].astype(float)
        df.index = df.index.normalize()  # strip time component → date-only alignment
        symbols[symbol] = df
    
    print(f"Loaded {len(symbols)} symbols")
    return symbols

# === METRICS LOADING ===
def load_metrics(metrics_dir='./merged_metrics/'):
    """Load Binance Vision metrics daily CSVs. Returns {} if not found."""
    metrics = {}
    files = glob.glob(os.path.join(metrics_dir, '*-metrics-daily.csv'))
    if not files:
        print("No metrics data found — skipping metrics signals")
        return metrics
    for fp in files:
        sym = os.path.basename(fp).replace('-metrics-daily.csv', '')
        df  = pd.read_csv(fp, index_col='date', parse_dates=True)
        df.index = df.index.normalize()  # strip time component → date-only alignment
        metrics[sym] = df
    print(f"Loaded metrics for {len(metrics)} symbols")
    return metrics

# === SIGNAL CALCULATION ===
def compute_signals(symbols, metrics=None):
    """Compute all signals per symbol, return panel dict {signal: DataFrame}."""
    print("Computing signals...")
    signal_names = [
        'reversal_1d', 'reversal_1w', 'momentum_30d',
        'volatility', 'liquidity', 'vol_compression', 'volume_compression',
        'taker_buy_contrarian',
        # New OHLCV signals
        'clv', 'taker_imbalance_momentum', 'intraday_range_ratio',
    ]
    # Add metrics signals if data available
    metrics_signal_names = []
    if metrics:
        metrics_signal_names = ['ls_contrarian', 'top_trader_contr', 'oi_price_signal']
        signal_names += metrics_signal_names

    signals = {s: {} for s in signal_names}
    
    for symbol, df in symbols.items():
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        taker_buy = df['taker_buy_volume']
        high_low = (high - low).replace(0, np.nan)
        pct = close.pct_change()
        
        signals['reversal_1d'][symbol] = -pct
        signals['reversal_1w'][symbol] = -close.pct_change(REVERSAL_LONG)
        signals['momentum_30d'][symbol] = close / close.shift(MOMENTUM_MED) - 1
        signals['volatility'][symbol] = -pct.rolling(VOL_WINDOW).std()
        signals['liquidity'][symbol] = (volume / high_low).rolling(LIQUIDITY_WIN).mean()
        signals['vol_compression'][symbol] = pct.rolling(VOL_SHORT).std() / pct.rolling(VOL_WINDOW).std()
        signals['volume_compression'][symbol] = volume / volume.rolling(VOLUME_WIN).mean()
        signals['taker_buy_contrarian'][symbol] = -(taker_buy / volume).rolling(TAKER_WIN).mean()

        # New Signal 1: Close Location Value (CLV)
        # (close - low) / (high - low), rolling 5d mean
        # Buying pressure proxy: 1 = closed at high, 0 = closed at low
        clv_raw = (close - low) / high_low
        signals['clv'][symbol] = clv_raw.rolling(5).mean()

        # New Signal 2: Taker Imbalance Momentum
        # Change in taker_buy_ratio over 5 days (delta of sentiment, not level)
        taker_ratio = taker_buy / volume
        signals['taker_imbalance_momentum'][symbol] = taker_ratio.diff(5)

        # New Signal 3: Intraday Range Ratio
        # (high - low) / close, 5d mean vs 30d mean
        # <1 = volatility compression -> potential breakout signal
        daily_range = high_low / close
        signals['intraday_range_ratio'][symbol] = daily_range.rolling(5).mean() / daily_range.rolling(30).mean()
    
    # Convert to DataFrames (date x symbol)
    for sig in list(signal_names):
        if sig not in metrics_signal_names:
            signals[sig] = pd.DataFrame(signals[sig])

    # === METRICS SIGNALS ===
    if metrics:
        ls_con   = {}
        tt_con   = {}
        oi_price = {}

        for sym, mdf in metrics.items():
            if sym not in symbols:
                continue
            close_s = symbols[sym]['close']

            # L/S contrarian: fade retail crowd (long when crowd is short)
            if 'ls_ratio_mean' in mdf.columns:
                ls_con[sym] = -mdf['ls_ratio_mean']

            # Top trader contrarian
            if 'top_trader_ls_mean' in mdf.columns:
                tt_con[sym] = -mdf['top_trader_ls_mean']

            # OI-price divergence: -(OI_pct_5d - price_pct_5d)
            # Needs enough history for 5d pct_change — early rows will be NaN (correct)
            if 'oi_value' in mdf.columns:
                oi_pct5    = mdf['oi_value'].pct_change(5)
                price_pct5 = close_s.reindex(mdf.index).pct_change(5)
                oi_price[sym] = -(oi_pct5 - price_pct5)

        if ls_con:
            signals['ls_contrarian']    = pd.DataFrame(ls_con)
        if tt_con:
            signals['top_trader_contr'] = pd.DataFrame(tt_con)
        if oi_price:
            signals['oi_price_signal']  = pd.DataFrame(oi_price)

        # Drop metrics keys that got no data
        for k in metrics_signal_names:
            if isinstance(signals.get(k), dict):
                del signals[k]

    print(f"Computed {len(signals)} signals"
          + (f" (incl. {len([k for k in signals if k in metrics_signal_names])} metrics)" if metrics else ""))
    return signals

# === PREPROCESSING ===
def preprocess(signals, symbols):
    """Z-score normalize signals, compute forward returns panel."""
    print("Preprocessing...")

    # Debug: print signal count and sample dates
    for sig_name, sig_df in signals.items():
        non_null = sig_df.notna().any(axis=1).sum()
        print(f"  {sig_name:<30} rows={len(sig_df)}  non-null-dates={non_null}")
    
    # Z-score each signal cross-sectionally
    for sig_name, sig_df in signals.items():
        def zscore(x):
            if x.std() == 0 or x.isna().all():
                return x * 0
            return ((x - x.mean()) / x.std()).clip(-CLIP_ZSCORE, CLIP_ZSCORE)
        signals[sig_name] = sig_df.apply(zscore, axis=1)
    
    # Build forward returns panel — use union of ALL signal dates (incl. metrics)
    all_dates = set()
    for df in symbols.values():
        all_dates.update(df.index)
    # Also include metrics signal dates so IC can be computed on those dates
    for sig_name, sig_df in signals.items():
        all_dates.update(sig_df.index)
    all_dates = sorted(all_dates)
    
    fwd_returns = {}
    for h in IC_DECAY_HORIZONS:
        fwd_returns[h] = pd.DataFrame(index=all_dates, columns=list(symbols.keys()), dtype=float)
        for sym, df in symbols.items():
            fwd = df['close'].pct_change(h).shift(-h)
            fwd_returns[h][sym] = fwd
    
    # Winsorize forward returns per date
    for h in IC_DECAY_HORIZONS:
        def winsor(x):
            if x.isna().all():
                return x
            lo, hi = x.quantile(WINSOR_PCTS[0]), x.quantile(WINSOR_PCTS[1])
            return x.clip(lo, hi)
        fwd_returns[h] = fwd_returns[h].apply(winsor, axis=1)
    
    return signals, fwd_returns

# === IC CALCULATION ===
def compute_ic(signals, fwd_returns, in_sample_dates=None, out_sample_dates=None):
    """Compute IC metrics for each signal and horizon."""
    print("Computing IC...")
    results = []
    
    for sig_name, sig_df in signals.items():
        for h in IC_DECAY_HORIZONS:
            fwd_df = fwd_returns[h]
            ics_spearman, ics_pearson = [], []
            ics_in, ics_out = [], []
            ics_sub1, ics_sub2 = [], []  # subsample stability: 2022-2023 vs 2024-2025
            
            sub1_cutoff = pd.Timestamp('2024-01-01')  # before = sub1, after = sub2
            
            for date in sig_df.index:
                if date not in fwd_df.index:
                    continue
                sig_row = sig_df.loc[date].dropna()
                fwd_row = fwd_df.loc[date].dropna()
                common = sig_row.index.intersection(fwd_row.index)
                
                if len(common) < MIN_SYMBOLS_PER_DATE:
                    continue
                
                x = sig_row[common].values
                y = fwd_row[common].values
                
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                
                ic_s, _ = spearmanr(x, y)
                ic_p, _ = pearsonr(x, y)
                ics_spearman.append(ic_s)
                ics_pearson.append(ic_p)
                
                if in_sample_dates and date in in_sample_dates:
                    ics_in.append(ic_s)
                elif out_sample_dates and date in out_sample_dates:
                    ics_out.append(ic_s)
                
                # Subsample stability
                if date < sub1_cutoff:
                    ics_sub1.append(ic_s)
                else:
                    ics_sub2.append(ic_s)
            
            if len(ics_spearman) == 0:
                continue
            
            ic_arr = np.array(ics_spearman)
            ic_mean = ic_arr.mean()
            ic_median = np.median(ic_arr)
            ic_std = ic_arr.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            
            # Standard t-stat (baseline)
            t_stat_std = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            
            # Newey-West t-stat (5 lags) — corrects for autocorrelation in IC series
            # Standard t-stat overstates significance by 20-30% when IC is autocorrelated
            try:
                nw_lags = 5
                ic_demeaned = ic_arr - ic_mean
                # HAC variance: Newey-West
                T = len(ic_arr)
                # Manual Newey-West: Var = gamma_0 + 2 * sum_l (1 - l/(nw_lags+1)) * gamma_l
                gamma_0 = np.sum(ic_demeaned ** 2) / T
                hac_var = gamma_0
                for lag in range(1, nw_lags + 1):
                    weight = 1.0 - lag / (nw_lags + 1)
                    gamma_l = np.sum(ic_demeaned[lag:] * ic_demeaned[:-lag]) / T
                    hac_var += 2 * weight * gamma_l
                hac_var = max(hac_var, 1e-12)  # floor at near-zero
                t_stat = ic_mean / np.sqrt(hac_var / T)
            except Exception:
                t_stat = t_stat_std  # fallback
            
            pct_pos = (ic_arr > 0).mean()
            
            ic_mean_in = np.mean(ics_in) if ics_in else np.nan
            ic_mean_out = np.mean(ics_out) if ics_out else np.nan
            ic_mean_sub1 = np.mean(ics_sub1) if ics_sub1 else np.nan   # 2022-2023
            ic_mean_sub2 = np.mean(ics_sub2) if ics_sub2 else np.nan   # 2024-2025
            
            results.append({
                'signal': sig_name,
                'horizon': h,
                'IC_mean': ic_mean,
                'IC_median': ic_median,
                'IC_std': ic_std,
                'ICIR': icir,
                't_stat': t_stat,
                't_stat_std': t_stat_std,
                'pct_positive': pct_pos,
                'turnover_mean': np.nan,  # filled later
                'IC_mean_in': ic_mean_in,
                'IC_mean_out': ic_mean_out,
                'IC_sub1_2022_2023': ic_mean_sub1,
                'IC_sub2_2024_2025': ic_mean_sub2,
                'n_days': len(ics_spearman)
            })
    
    return pd.DataFrame(results)

# === TURNOVER ===
def compute_turnover(signals, ic_df):
    """Compute factor turnover for each signal."""
    print("Computing turnover...")
    
    for sig_name in ic_df['signal'].unique():
        sig_df = signals[sig_name]
        turnovers = []
        
        dates = sig_df.index.tolist()
        for i in range(1, len(dates)):
            d1, d2 = dates[i-1], dates[i]
            s1 = sig_df.loc[d1].dropna()
            s2 = sig_df.loc[d2].dropna()
            common = s1.index.intersection(s2.index)
            
            if len(common) < MIN_SYMBOLS_PER_DATE:
                continue
            
            r1 = rankdata(s1[common].values)
            r2 = rankdata(s2[common].values)
            to = np.mean(np.abs(r1 - r2)) / len(common)
            turnovers.append(to)
        
        avg_to = np.mean(turnovers) if turnovers else np.nan
        ic_df.loc[(ic_df['signal'] == sig_name), 'turnover_mean'] = avg_to
    
    return ic_df

# === IN/OUT SAMPLE ===
def split_dates(signals):
    """Split dates into in-sample and out-of-sample."""
    all_dates = set()
    for sig_df in signals.values():
        all_dates.update(sig_df.index)
    all_dates = sorted(all_dates)
    
    split_idx = int(len(all_dates) * IN_SAMPLE_RATIO)
    in_sample = set(all_dates[:split_idx])
    out_sample = set(all_dates[split_idx:])
    
    return in_sample, out_sample, all_dates[0], all_dates[-1]

# === VISUALIZATION ===
def create_charts(ic_df):
    """Create IC visualization charts."""
    print("Creating charts...")
    
    # IC Mean at horizon=1
    h1 = ic_df[ic_df['horizon'] == 1].copy()
    h1 = h1.sort_values('IC_mean', key=lambda x: abs(x), ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in h1['IC_mean']]
    y_pos = range(len(h1))
    ax.barh(y_pos, h1['IC_mean'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(h1['signal'])
    ax.set_xlabel('IC Mean')
    ax.set_title('IC Mean at Horizon=1')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(0.02, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(-0.02, color='gray', linestyle='--', linewidth=0.5)
    
    for i, (ic, t, t_std) in enumerate(zip(h1['IC_mean'], h1['t_stat'], h1['t_stat_std'])):
        ax.text(ic, i, f' {ic:.3f} (NW t={t:.2f}, std t={t_std:.2f})', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('ic_chart.png', dpi=150)
    plt.close()
    
    # Subsample stability chart (2022-2023 vs 2024-2025)
    h1_sub = ic_df[ic_df['horizon'] == 1].copy()
    h1_sub = h1_sub.dropna(subset=['IC_sub1_2022_2023', 'IC_sub2_2024_2025'])
    h1_sub = h1_sub.sort_values('IC_mean', key=lambda x: abs(x), ascending=True)
    
    if len(h1_sub) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(h1_sub))
        width = 0.35
        ax.barh(y_pos - width/2, h1_sub['IC_sub1_2022_2023'], width, label='2022-2023', color='steelblue', alpha=0.8)
        ax.barh(y_pos + width/2, h1_sub['IC_sub2_2024_2025'], width, label='2024-2025', color='darkorange', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(h1_sub['signal'])
        ax.set_xlabel('IC Mean')
        ax.set_title('Subsample Stability: 2022-2023 vs 2024-2025 (Horizon=1)')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig('ic_subsample_chart.png', dpi=150)
        plt.close()
    
    # IC Decay
    fig, ax = plt.subplots(figsize=(10, 6))
    for sig in ic_df['signal'].unique():
        sub = ic_df[ic_df['signal'] == sig].sort_values('horizon')
        ax.plot(sub['horizon'], sub['IC_mean'], marker='o', label=sig)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('IC Mean')
    ax.set_title('IC Decay by Horizon')
    ax.legend(loc='best', fontsize=8)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('ic_decay_chart.png', dpi=150)
    plt.close()

# === SAVE OUTPUTS ===
def save_outputs(ic_df, n_symbols, date_start, date_end, in_sample, out_sample):
    """Save all outputs."""
    print("Saving outputs...")
    ic_df.to_csv('ic_results.csv', index=False)
    
    # Summary
    top_icir = ic_df.nlargest(3, 'ICIR')[['signal', 'ICIR', 't_stat']]
    sig_tstat = ic_df[ic_df['t_stat'] > 2.0]['signal'].unique()
    
    # Check OOS drop
    warnings = []
    for sig in ic_df['signal'].unique():
        sub = ic_df[ic_df['signal'] == sig]
        ic_in = sub['IC_mean_in'].mean()
        ic_out = sub['IC_mean_out'].mean()
        if not np.isnan(ic_in) and not np.isnan(ic_out) and abs(ic_in) > 0:
            drop = (abs(ic_in) - abs(ic_out)) / abs(ic_in)
            if drop > 0.5:
                warnings.append(f"WARNING: {sig} OOS IC dropped {drop*100:.1f}% vs in-sample")
    
    with open('ic_summary.txt', 'w') as f:
        f.write(f"Total symbols loaded: {n_symbols}\n")
        f.write(f"Date range: {date_start} to {date_end}\n")
        f.write(f"In-sample dates: {len(in_sample)}\n")
        f.write(f"Out-of-sample dates: {len(out_sample)}\n\n")
        
        f.write("=== IC at Horizon=1 (sorted by |NW t-stat|) ===\n")
        h1 = ic_df[ic_df['horizon'] == 1].copy()
        h1['abs_t'] = h1['t_stat'].abs()
        h1 = h1.sort_values('abs_t', ascending=False)
        f.write(f"{'Signal':<30} {'IC_mean':>8} {'IC_median':>10} {'NW_t':>8} {'std_t':>8} {'ICIR':>7} {'%pos':>6}\n")
        f.write("-" * 80 + "\n")
        for _, row in h1.iterrows():
            sig_flag = "✅" if abs(row['t_stat']) > 2.0 else ("⚠️" if abs(row['t_stat']) > 1.5 else "❌")
            f.write(f"{row['signal']:<30} {row['IC_mean']:>8.4f} {row['IC_median']:>10.4f} "
                    f"{row['t_stat']:>8.2f} {row['t_stat_std']:>8.2f} {row['ICIR']:>7.3f} "
                    f"{row['pct_positive']:>6.2%}  {sig_flag}\n")
        
        f.write("\n=== Subsample Stability (Horizon=1): 2022-2023 vs 2024-2025 ===\n")
        f.write(f"{'Signal':<30} {'sub1_IC':>10} {'sub2_IC':>10} {'sign_flip':>10}\n")
        f.write("-" * 65 + "\n")
        for _, row in h1.iterrows():
            s1 = row['IC_sub1_2022_2023']
            s2 = row['IC_sub2_2024_2025']
            if pd.isna(s1) or pd.isna(s2):
                continue
            flip = "⚠️ FLIP" if np.sign(s1) != np.sign(s2) else "✅ stable"
            f.write(f"{row['signal']:<30} {s1:>10.4f} {s2:>10.4f} {flip:>10}\n")
        
        f.write("\n=== Top 3 by ICIR (all horizons) ===\n")
        top_icir = ic_df.nlargest(3, 'ICIR')[['signal', 'horizon', 'ICIR', 't_stat']]
        for _, row in top_icir.iterrows():
            f.write(f"  {row['signal']} h={int(row['horizon'])}: ICIR={row['ICIR']:.3f}, NW t={row['t_stat']:.2f}\n")
        
        if warnings:
            f.write("\n" + "\n".join(warnings) + "\n")

# === MAIN ===
if __name__ == '__main__':
    symbols = load_data()
    metrics = load_metrics()
    signals = compute_signals(symbols, metrics)
    signals, fwd_returns = preprocess(signals, symbols)
    in_sample, out_sample, date_start, date_end = split_dates(signals)
    
    ic_df = compute_ic(signals, fwd_returns, in_sample, out_sample)
    ic_df = compute_turnover(signals, ic_df)
    
    create_charts(ic_df)
    save_outputs(ic_df, len(symbols), date_start, date_end, in_sample, out_sample)
    
    print("Done. Outputs: ic_results.csv, ic_chart.png, ic_decay_chart.png, ic_summary.txt")
