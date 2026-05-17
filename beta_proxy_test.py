#!/usr/bin/env python3
"""Test if composite signal is alpha or beta proxy."""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# === LOAD DATA ===
def load_btc_monthly(data_dir='./merged_data/'):
    """Load BTC daily data and compute monthly returns."""
    fp = os.path.join(data_dir, 'BTCUSDT-1d-full.csv')
    df = pd.read_csv(fp)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('date').sort_index()
    df['month'] = df.index.to_period('M')
    
    monthly = df.groupby('month').agg(
        start_close=('close', 'first'),
        end_close=('close', 'last')
    )
    monthly['btc_return'] = (monthly['end_close'] / monthly['start_close'] - 1)
    return monthly['btc_return']

def compute_daily_ic(scores_file='./composite/composite_scores_meta.csv', 
                     data_dir='./merged_data/'):
    """Compute daily IC from composite scores (wide format: symbols as columns)."""
    scores = pd.read_csv(scores_file)
    scores['date'] = pd.to_datetime(scores['date'])
    scores = scores.set_index('date')
    
    # Load all close prices
    prices = {}
    for fp in os.listdir(data_dir):
        if not fp.endswith('.csv'):
            continue
        symbol = fp.replace('.csv', '').replace('-1d-full', '')
        df = pd.read_csv(os.path.join(data_dir, fp))
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('date').sort_index()
        prices[symbol] = df['close']
    
    # Compute IC per date
    dates = scores.index
    ic_records = []
    
    for i in range(len(dates) - 1):
        dt = dates[i]
        next_dt = dates[i + 1]
        
        scores_row = scores.loc[dt]
        ic_vals = []
        
        for symbol in scores.columns:
            score_val = scores_row[symbol]
            if pd.isna(score_val) or symbol not in prices:
                continue
            try:
                ret_today = prices[symbol].get(dt)
                ret_next = prices[symbol].get(next_dt)
                if ret_today is None or ret_next is None or pd.isna(ret_today) or pd.isna(ret_next):
                    continue
                fwd_ret = ret_next / ret_today - 1
                ic_vals.append((score_val, fwd_ret))
            except:
                continue
        
        if len(ic_vals) >= 5:
            scores_arr = np.array([x[0] for x in ic_vals])
            rets_arr = np.array([x[1] for x in ic_vals])
            p1, p99 = np.percentile(rets_arr, [1, 99])
            rets_arr = np.clip(rets_arr, p1, p99)
            ic, _ = spearmanr(scores_arr, rets_arr)
            if not np.isnan(ic):
                ic_records.append({'date': dt, 'IC': ic})
    
    return pd.DataFrame(ic_records)

def main():
    print("=" * 60)
    print("BETA PROXY TEST: Composite Signal vs BTC Return")
    print("=" * 60)
    
    print("\n[1] Loading BTC monthly returns...")
    btc_monthly = load_btc_monthly()
    print(f"    Found {len(btc_monthly)} months of BTC data")
    print(f"    Range: {btc_monthly.index[0]} to {btc_monthly.index[-1]}")
    
    print("\n[2] Computing daily IC from composite scores...")
    daily_ic = compute_daily_ic()
    daily_ic['month'] = pd.to_datetime(daily_ic['date']).dt.to_period('M')
    print(f"    Found {len(daily_ic)} daily IC values")
    
    print("\n[3] Aggregating to monthly average IC...")
    monthly_ic = daily_ic.groupby('month')['IC'].mean()
    print(f"    Found {len(monthly_ic)} months with IC data")
    
    common_months = btc_monthly.index.intersection(monthly_ic.index)
    btc_aligned = btc_monthly[common_months]
    ic_aligned = monthly_ic[common_months]
    print(f"    Common months: {len(common_months)}")
    
    print("\n[4] Computing correlations...")
    pearson_corr, pearson_p = pearsonr(btc_aligned.values, ic_aligned.values)
    spearman_corr, spearman_p = spearmanr(btc_aligned.values, ic_aligned.values)
    
    print(f"\n    Pearson correlation:  {pearson_corr:+.4f} (p={pearson_p:.4f})")
    print(f"    Spearman correlation: {spearman_corr:+.4f} (p={spearman_p:.4f})")
    
    print("\n[5] Conditional IC analysis...")
    pos_mask = btc_aligned > 0
    neg_mask = btc_aligned <= 0
    
    ic_when_btc_up = ic_aligned[pos_mask].mean()
    ic_when_btc_down = ic_aligned[neg_mask].mean()
    
    print(f"    Months BTC positive: {pos_mask.sum()}")
    print(f"    Months BTC negative: {neg_mask.sum()}")
    print(f"    Avg IC when BTC up:   {ic_when_btc_up:+.4f}")
    print(f"    Avg IC when BTC down: {ic_when_btc_down:+.4f}")
    print(f"    Difference:           {ic_when_btc_up - ic_when_btc_down:+.4f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    abs_corr = abs(pearson_corr)
    if abs_corr > 0.35:
        evidence = "EVIDENCE: Signal kemungkinan BETA PROXY"
        detail = f"Correlation {pearson_corr:+.3f} > 0.35 → IC strongly related to market direction"
    elif abs_corr < 0.15:
        evidence = "EVIDENCE: Signal kemungkinan GENUINE ALPHA"
        detail = f"Correlation {pearson_corr:+.3f} < 0.15 → IC independent of market regime"
    else:
        evidence = "INCONCLUSIVE: Perlu analisis lebih dalam"
        detail = f"Correlation {pearson_corr:+.3f} in range [0.15, 0.35]"
    
    print(f"\n{evidence}")
    print(f"{detail}")
    
    if ic_when_btc_up > ic_when_btc_down + 0.02:
        print(f"\nAdditional: IC lebih tinggi saat BTC naik → pro-cyclical signal")
    elif ic_when_btc_down > ic_when_btc_up + 0.02:
        print(f"\nAdditional: IC lebih tinggi saat BTC turun → counter-cyclical signal")
    else:
        print(f"\nAdditional: IC stabil across market regimes → regime-independent")
    
    print("\n[6] Generating scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = [m.year for m in common_months]
    unique_years = sorted(set(years))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))
    year_to_color = dict(zip(unique_years, colors))
    
    for year in unique_years:
        mask = np.array([m.year == year for m in common_months])
        ax.scatter(btc_aligned[mask], ic_aligned[mask], 
                   color=year_to_color[year], label=str(year), 
                   alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
    
    z = np.polyfit(btc_aligned, ic_aligned, 1)
    p = np.poly1d(z)
    x_line = np.linspace(btc_aligned.min(), btc_aligned.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={pearson_corr:.2f})')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Monthly BTC Return', fontsize=12)
    ax.set_ylabel('Monthly Average IC', fontsize=12)
    ax.set_title('Composite IC vs BTC Return (Beta Proxy Test)', fontsize=14, fontweight='bold')
    ax.legend(title='Year', loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('beta_proxy_test.png', dpi=150, bbox_inches='tight')
    print("    Saved: beta_proxy_test.png")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BTC months:        {len(btc_aligned)}")
    print(f"IC range:          [{ic_aligned.min():.3f}, {ic_aligned.max():.3f}]")
    print(f"BTC return range:  [{btc_aligned.min():.2%}, {btc_aligned.max():.2%}]")
    print(f"Pearson corr:      {pearson_corr:+.4f}")
    print(f"Spearman corr:     {spearman_corr:+.4f}")
    print(f"IC (BTC up):       {ic_when_btc_up:+.4f}")
    print(f"IC (BTC down):     {ic_when_btc_down:+.4f}")
    print(f"Conclusion:        {evidence.split(':')[1].strip()}")
    print("=" * 60)

if __name__ == '__main__':
    main()
