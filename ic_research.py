#!/usr/bin/env python3
"""Cross-sectional factor IC analysis for crypto using local OHLCV data."""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
import matplotlib.pyplot as plt

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
MIN_SYMBOLS_PER_DATE = 5
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
        symbol = os.path.basename(fp).replace('.csv', '')
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('date').sort_index()
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['quote_volume'] = df['quote_volume'].astype(float)
        df['taker_buy_volume'] = df['taker_buy_volume'].astype(float)
        symbols[symbol] = df
    
    print(f"Loaded {len(symbols)} symbols")
    return symbols

# === SIGNAL CALCULATION ===
def compute_signals(symbols):
    """Compute all signals per symbol, return panel dict {signal: DataFrame}."""
    print("Computing signals...")
    signal_names = [
        'reversal_1d', 'reversal_1w', 'momentum_30d', 'momentum_90d',
        'volatility', 'liquidity', 'vol_compression', 'volume_compression',
        'price_to_high', 'taker_buy_contrarian', 'quote_liquidity'
    ]
    signals = {s: {} for s in signal_names}
    
    for symbol, df in symbols.items():
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        quote_volume = df['quote_volume']
        taker_buy = df['taker_buy_volume']
        high_low = (high - low).replace(0, np.nan)
        pct = close.pct_change()
        
        signals['reversal_1d'][symbol] = -pct
        signals['reversal_1w'][symbol] = -close.pct_change(REVERSAL_LONG)
        signals['momentum_30d'][symbol] = close / close.shift(MOMENTUM_MED) - 1
        signals['momentum_90d'][symbol] = close / close.shift(MOMENTUM_LONG) - 1
        signals['volatility'][symbol] = -pct.rolling(VOL_WINDOW).std()
        signals['liquidity'][symbol] = (volume / high_low).rolling(LIQUIDITY_WIN).mean()
        signals['vol_compression'][symbol] = pct.rolling(VOL_SHORT).std() / pct.rolling(VOL_WINDOW).std()
        signals['volume_compression'][symbol] = volume / volume.rolling(VOLUME_WIN).mean()
        signals['price_to_high'][symbol] = close / close.rolling(PRICE_HIGH_WIN).max()
        signals['taker_buy_contrarian'][symbol] = -(taker_buy / volume).rolling(TAKER_WIN).mean()
        signals['quote_liquidity'][symbol] = (quote_volume / high_low).rolling(LIQUIDITY_WIN).mean()
    
    # Convert to DataFrames (date x symbol)
    for sig in signal_names:
        signals[sig] = pd.DataFrame(signals[sig])
    
    print(f"Computed {len(signal_names)} signals")
    return signals

# === PREPROCESSING ===
def preprocess(signals, symbols):
    """Z-score normalize signals, compute forward returns panel."""
    print("Preprocessing...")
    
    # Z-score each signal cross-sectionally
    for sig_name, sig_df in signals.items():
        def zscore(x):
            if x.std() == 0 or x.isna().all():
                return x * 0
            return ((x - x.mean()) / x.std()).clip(-CLIP_ZSCORE, CLIP_ZSCORE)
        signals[sig_name] = sig_df.apply(zscore, axis=1)
    
    # Build forward returns panel
    all_dates = set()
    for df in symbols.values():
        all_dates.update(df.index)
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
            
            if len(ics_spearman) == 0:
                continue
            
            ic_arr = np.array(ics_spearman)
            ic_mean = ic_arr.mean()
            ic_std = ic_arr.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            pct_pos = (ic_arr > 0).mean()
            
            ic_mean_in = np.mean(ics_in) if ics_in else np.nan
            ic_mean_out = np.mean(ics_out) if ics_out else np.nan
            
            results.append({
                'signal': sig_name,
                'horizon': h,
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'ICIR': icir,
                't_stat': t_stat,
                'pct_positive': pct_pos,
                'turnover_mean': np.nan,  # filled later
                'IC_mean_in': ic_mean_in,
                'IC_mean_out': ic_mean_out,
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
    
    for i, (ic, t) in enumerate(zip(h1['IC_mean'], h1['t_stat'])):
        ax.text(ic, i, f' {ic:.3f} (t={t:.2f})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ic_chart.png', dpi=150)
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
        f.write("Top 3 signals by ICIR:\n")
        for _, row in top_icir.iterrows():
            f.write(f"  {row['signal']}: ICIR={row['ICIR']:.3f}, t={row['t_stat']:.2f}\n")
        f.write(f"\nSignals with t_stat > 2.0: {', '.join(sig_tstat) if len(sig_tstat) else 'None'}\n")
        if warnings:
            f.write("\n" + "\n".join(warnings) + "\n")

# === MAIN ===
if __name__ == '__main__':
    symbols = load_data()
    signals = compute_signals(symbols)
    signals, fwd_returns = preprocess(signals, symbols)
    in_sample, out_sample, date_start, date_end = split_dates(signals)
    
    ic_df = compute_ic(signals, fwd_returns, in_sample, out_sample)
    ic_df = compute_turnover(signals, ic_df)
    
    create_charts(ic_df)
    save_outputs(ic_df, len(symbols), date_start, date_end, in_sample, out_sample)
    
    print("Done. Outputs: ic_results.csv, ic_chart.png, ic_decay_chart.png, ic_summary.txt")
