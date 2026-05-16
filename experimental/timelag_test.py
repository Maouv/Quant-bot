#!/usr/bin/env python3
"""Time lag analysis: test if IC has forward-looking bias or price-in issues."""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime

# === PARAMETERS ===
NW_LAGS = 5
MIN_SYMBOLS = 5

# === NEWey-WEST T-STAT ===
def newey_west_tstat(ic_arr):
    """Compute Newey-West adjusted t-stat (5 lags)."""
    ic_arr = np.array(ic_arr)
    ic_arr = ic_arr[~np.isnan(ic_arr)]
    if len(ic_arr) < 10:
        return np.nan, np.nan
    
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1)
    
    if ic_std == 0:
        return ic_mean, 0
    
    # Manual Newey-West
    T = len(ic_arr)
    ic_demeaned = ic_arr - ic_mean
    gamma_0 = np.sum(ic_demeaned ** 2) / T
    hac_var = gamma_0
    
    for lag in range(1, NW_LAGS + 1):
        weight = 1.0 - lag / (NW_LAGS + 1)
        gamma_l = np.sum(ic_demeaned[lag:] * ic_demeaned[:-lag]) / T
        hac_var += 2 * weight * gamma_l
    
    hac_var = max(hac_var, 1e-12)
    t_stat = ic_mean / np.sqrt(hac_var / T)
    
    return ic_mean, t_stat

# === LOAD DATA ===
def load_returns(data_dir='./merged_data/'):
    """Load close prices and compute daily returns."""
    import glob
    returns = {}
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for fp in csv_files:
        symbol = os.path.basename(fp).replace('.csv', '').replace('-1d-full', '')
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('date').sort_index()
        df['ret'] = df['close'].pct_change()
        returns[symbol] = df['ret']
    
    # Combine into DataFrame
    ret_df = pd.DataFrame(returns)
    ret_df.index = ret_df.index.normalize()
    
    return ret_df

# === MAIN ===
def main():
    print("=== TIME LAG ANALYSIS ===\n")
    
    # Load composite scores
    print("Loading composite scores...")
    scores_df = pd.read_csv('./composite/composite_scores_meta.csv', index_col=0, parse_dates=True)
    print(f"  Scores shape: {scores_df.shape}")
    
    # Load returns
    print("Loading returns...")
    ret_df = load_returns()
    print(f"  Returns shape: {ret_df.shape}")
    
    # Align dates
    common_dates = scores_df.index.intersection(ret_df.index)
    scores_df = scores_df.loc[common_dates]
    ret_df = ret_df.loc[common_dates]
    print(f"  Common dates: {len(common_dates)}")
    
    # Compute ICs for different lags
    ic_results = {
        'same_day': [],  # signal vs same day return
        'lag0': [],      # signal vs next day return (current impl)
        'lag1': [],      # signal from t-1 vs next day return
        'lag2': [],      # signal from t-2 vs next day return
    }
    
    ic_by_year = {year: {'lag0': [], 'lag1': [], 'lag2': []} for year in range(2022, 2026)}
    
    print("\nComputing ICs...")
    for i in range(2, len(scores_df) - 1):  # Need at least 2 days of history
        date = scores_df.index[i]
        year = date.year
        
        # Get signals at different lags
        signal_lag0 = scores_df.iloc[i].dropna()
        signal_lag1 = scores_df.iloc[i-1].dropna()
        signal_lag2 = scores_df.iloc[i-2].dropna()
        
        # Get returns
        ret_same_day = ret_df.iloc[i].dropna()    # return hari ini
        ret_next_day = ret_df.iloc[i+1].dropna()  # return besok
        
        # Common symbols
        common0 = signal_lag0.index.intersection(ret_next_day.index)
        common1 = signal_lag1.index.intersection(ret_next_day.index)
        common2 = signal_lag2.index.intersection(ret_next_day.index)
        common_same = signal_lag0.index.intersection(ret_same_day.index)
        
        # Compute ICs if enough symbols
        if len(common0) >= MIN_SYMBOLS:
            ic, _ = spearmanr(signal_lag0[common0], ret_next_day[common0])
            if not np.isnan(ic):
                ic_results['lag0'].append(ic)
                ic_by_year[year]['lag0'].append(ic)
        
        if len(common1) >= MIN_SYMBOLS:
            ic, _ = spearmanr(signal_lag1[common1], ret_next_day[common1])
            if not np.isnan(ic):
                ic_results['lag1'].append(ic)
                ic_by_year[year]['lag1'].append(ic)
        
        if len(common2) >= MIN_SYMBOLS:
            ic, _ = spearmanr(signal_lag2[common2], ret_next_day[common2])
            if not np.isnan(ic):
                ic_results['lag2'].append(ic)
                ic_by_year[year]['lag2'].append(ic)
        
        if len(common_same) >= MIN_SYMBOLS:
            ic, _ = spearmanr(signal_lag0[common_same], ret_same_day[common_same])
            if not np.isnan(ic):
                ic_results['same_day'].append(ic)
    
    # Compute metrics
    metrics = {}
    for name, ics in ic_results.items():
        ics = np.array(ics)
        if len(ics) == 0:
            metrics[name] = {'mean': 0, 'std': 0, 'icir': 0, 't_stat': 0, 'pct_pos': 0}
            continue
        
        ic_mean, t_stat = newey_west_tstat(ics)
        ic_std = np.std(ics, ddof=1)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        pct_pos = (ics > 0).mean() * 100
        
        metrics[name] = {
            'mean': ic_mean,
            'std': ic_std,
            'icir': icir,
            't_stat': t_stat,
            'pct_pos': pct_pos
        }
    
    # Compute yearly metrics
    year_metrics = {}
    for year in range(2022, 2026):
        year_metrics[year] = {}
        for lag in ['lag0', 'lag1', 'lag2']:
            ics = ic_by_year[year][lag]
            if len(ics) >= 10:
                mean, _ = newey_west_tstat(ics)
            else:
                mean = np.nan
            year_metrics[year][lag] = mean
    
    # Generate verdict
    ic_lag0 = metrics['lag0']['mean']
    ic_lag1 = metrics['lag1']['mean']
    ic_lag2 = metrics['lag2']['mean']
    ic_same = metrics['same_day']['mean']
    
    verdict_lines = []
    
    # Time lag verdict
    if ic_lag0 > ic_lag1:
        verdict_lines.append("✅ NO TIME LAG — signal fresh, trade timing OK")
    elif ic_lag1 > ic_lag0 * 1.1:
        verdict_lines.append("🚨 TIME LAG CONFIRMED — use lagged signal")
    else:
        verdict_lines.append("⚠️ MARGINAL — difference small, monitor")
    
    # Intraday price-in verdict
    if abs(ic_same) > abs(ic_lag0) * 0.5:
        verdict_lines.append("⚠️ INTRADAY PRICE-IN DETECTED")
    else:
        verdict_lines.append("✅ NO INTRADAY PRICE-IN")
    
    # Format output
    output = []
    output.append("=== TIME LAG ANALYSIS ===\n")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.append(f"Total observations: {len(ic_results['lag0'])}\n\n")
    
    output.append("IC SUMMARY:\n")
    output.append(f"IC_same_day: mean={metrics['same_day']['mean']:+.4f}  ICIR={metrics['same_day']['icir']:+.3f}  t={metrics['same_day']['t_stat']:+.2f}  (signal vs same day return)\n")
    output.append(f"IC_lag0:     mean={metrics['lag0']['mean']:+.4f}  ICIR={metrics['lag0']['icir']:+.3f}  t={metrics['lag0']['t_stat']:+.2f}  (current implementation)\n")
    output.append(f"IC_lag1:     mean={metrics['lag1']['mean']:+.4f}  ICIR={metrics['lag1']['icir']:+.3f}  t={metrics['lag1']['t_stat']:+.2f}  (signal shifted 1 day)\n")
    output.append(f"IC_lag2:     mean={metrics['lag2']['mean']:+.4f}  ICIR={metrics['lag2']['icir']:+.3f}  t={metrics['lag2']['t_stat']:+.2f}  (signal shifted 2 days)\n\n")
    
    output.append("BY YEAR:\n")
    output.append(f"{'Year':<6} {'lag0_IC':>10} {'lag1_IC':>10} {'lag2_IC':>10}\n")
    for year in range(2022, 2026):
        output.append(f"{year:<6} {year_metrics[year]['lag0']:+10.4f} {year_metrics[year]['lag1']:+10.4f} {year_metrics[year]['lag2']:+10.4f}\n")
    output.append("\n")
    
    output.append("VERDICT:\n")
    for line in verdict_lines:
        output.append(f"  {line}\n")
    
    output.append("\n=== END ===\n")
    
    # Print to console
    print(''.join(output))
    
    # Save to file
    os.makedirs('./experimental/timelag_test', exist_ok=True)
    with open('./experimental/timelag_test/timelag_results.txt', 'w') as f:
        f.write(''.join(output))
    
    print(f"\nResults saved to experimental/timelag_test/timelag_results.txt")

if __name__ == '__main__':
    main()
