# Opus Validation Summary — Crypto IC Factor Research
> Semua jawaban dari Claude Opus berdasarkan sesi validasi framework IC analysis

---

## Q1 — Methodology Validation

**Pertanyaan:** Apakah framework cross-sectional IC analysis untuk crypto futures sudah sound?

### Verdict: ✅ Sound

Spearman rank IC cross-sectional adalah pendekatan standar di quantitative research (Qian, Hua & Sorensen 2007) dan applicable untuk crypto. Pilihan yang sudah benar:

- Cross-sectional z-score + winsorization → correct untuk fat tails crypto
- Spearman over Pearson → robust terhadap outlier
- ICIR dan t-stat sebagai primary metrics → standard
- In-sample/out-of-sample split → good discipline
- Liu et al. (2022) framing → appropriate

### Critical Flaws yang Ditemukan

| Issue | Severity | Action |
|---|---|---|
| Survivorship bias (26 symbols aktif, LUNA/FTT excluded) | Medium | Dokumentasikan eksplisit, tidak fatal untuk signal discovery |
| N=26 cross-sectional noise | Medium | T=1000 hari menyelamatkan — law of large numbers |
| Volatility ↔ liquidity mungkin berkorelasi tinggi | High | Harus dicek correlation matrix |
| Tidak ada market-cap control | Medium | Pertimbangkan residualize vs log(avg_volume) |
| Lookahead bias risk | High | Double-check timestamp alignment |

### Sinyal Tambahan yang Direkomendasikan (dari OHLCV saja)

| Sinyal | Formula | Rationale |
|---|---|---|
| Intraday range ratio | `(high-low)/close` normalized 5d vs 30d | Breakout detection |
| Close Location Value (CLV) | `(close-low)/(high-low)` rolling 5d mean | Buying pressure proxy, **paling promising** |
| Taker imbalance momentum | `Δ(taker_buy_ratio)` over 5d | Change in sentiment, bukan level |
| Volume-weighted return | `Σ(ret×vol)/Σ(vol)` over 5d | Volume confirm direction |
| Reversal 2w | `-pct_change(14)` | Gap antara 1w dan 30d |

### Momentum Decay Pattern

Confirmed sesuai literatur:
- **Jegadeesh & Titman (1993):** momentum returns peak di 3-12 bulan formation
- **Liu et al. (2022):** explicit mention very short-term returns exhibit reversal, not momentum
- Pattern kita (reversal dominates h=1, momentum emerges h=10-20) adalah **textbook transition** dari mean-reversion ke trend-following

---

## Q2 — Universe Size N=26

**Pertanyaan:** Apakah 26 symbols cukup untuk cross-sectional IC yang reliable?

### Verdict: ✅ Acceptable (dengan caveats)

$$SE(\rho) \approx \frac{1}{\sqrt{N-1}} = \frac{1}{\sqrt{25}} = 0.20$$

Single daily IC estimate sangat noisy (±0.20). Tapi:

$$t = \frac{\overline{IC}}{\sigma_{IC}/\sqrt{T}} \quad \text{dengan } T \approx 1000$$

Law of large numbers rescue — averaging 1000 observasi drives SE of mean ke ~0.01.

### Yang N=26 Tidak Bisa Dilakukan
- Individual daily IC tidak reliable (jangan baca single day)
- Tidak bisa quintile sort (5 aset per quintile — terlalu sedikit)
- Sinyal lemah (IC < 0.02) akan terkubur noise
- Borderline signals (t=1.90, t=1.84) mungkin significant kalau N=40-50

### Time-Series IC — Lebih Buruk, Bukan Lebih Baik

| Alasan | Detail |
|---|---|
| Confounds cross-sectional & time-series | Signal bisa punya zero cross-sectional IC tapi strong time-series IC |
| Autocorrelation destroys effective T | Signal persistent → jauh lebih sedikit effective DoF |
| Bukan yang dipakai literatur | Liu et al., Fama-French semua pakai cross-sectional |
| Heterogeneity across symbols | BTC dan small-cap alt dynamics sangat berbeda |

### Robustness Checks yang Harus Ditambah

1. **Newey-West t-stats** — standard t-stat mungkin overstated 20-30% karena autocorrelation IC values. Gunakan `statsmodels` dengan ~5 lags
2. **Median IC** alongside mean — less sensitive to outlier days
3. **Subsample stability** — split 2022-2023 vs 2024-2025, confirm IC sign konsisten
4. **MIN_SYMBOLS naik ke 15** — dari 5, untuk lebih reliable daily IC estimates di early dates

---

## Q3 — Transaction Cost Analysis

**Pertanyaan:** Apakah reversal alpha survive setelah transaction costs?

### Break-even IC Calculation

$$IC_{break-even} = \frac{turnover \times cost_{RT}}{2 \times \sigma_{cs}}$$

**Asumsi:**
- Taker fee: 0.04% per side × 2 = 0.08% round-trip
- Slippage: ~0.03% round-trip
- Total cost: ~0.11% round-trip
- Cross-sectional dispersion: 4%/day

| Signal | IC | Turnover | Cost Drag/day | Break-even IC | Margin |
|---|---|---|---|---|---|
| reversal_1d | 0.018 | 0.33 | 0.036%/day | 0.0045 | **4×** |
| reversal_1w | 0.023 | 0.13 | 0.014%/day | 0.0018 | **13×** |

**Annual cost drag:**
- reversal_1d: ~13.3% annualized
- reversal_1w: ~5.2% annualized

### Net Daily Return Estimate

$$E[r_{daily}] \approx 2 \times IC \times \sigma_{cs}$$

| Signal | Gross Daily | Cost Drag | Net Daily |
|---|---|---|---|
| reversal_1d | 0.144% | 0.036% | **+0.108%** |
| reversal_1w | 0.184% | 0.014% | **+0.170%** |

### Verdict

✅ Kedua survive transaction costs.

**reversal_1w jauh superior** — IC lebih tinggi, turnover lebih rendah, 3× better cost-adjusted return.

### Practical Improvements

1. **Maker orders** → potong fee 50% (0.04% → 0.02% per side)
2. **Rebalance threshold** → hanya trade kalau rank berubah > X → kurangi turnover 30-40%
3. **Blend reversal_1d ke composite** — jangan trade standalone

### Warning: Capacity Risk

RNDRUSDT dan TIAUSDT order book tipis → kena market impact sebelum fee jadi masalah. Weight by liquidity saat portfolio construction.

---

## Q4 — Signal Correlation & Redundancy

**Pertanyaan:** Signal mana yang redundant?

### Correlation Matrix Hasil

```
=== PAIRS WITH |CORR| > 0.5 ===
reversal_1w   <-> price_to_high:  -0.665
momentum_30d  <-> momentum_90d:    0.557
momentum_30d  <-> price_to_high:   0.848
liquidity     <-> quote_liquidity: 0.822
```

**Temuan penting:** Volatility ↔ liquidity hanya **0.280** — tidak redundant seperti yang dikhawatirkan. Keduanya tetap masuk composite.

### Signal yang Di-DROP

| Drop | Keep | Correlation | Alasan |
|---|---|---|---|
| `quote_liquidity` | `liquidity` | 0.822 | Sama persis, beda unit saja |
| `momentum_90d` | `momentum_30d` | 0.557 | 30d lebih actionable |
| `price_to_high` | `momentum_30d` | 0.848 | Hampir identik |

### Cutoff Framework

| Correlation | Action |
|---|---|
| > 0.80 | Always drop one |
| 0.50 – 0.80 | Test incremental IC, conditional drop |
| 0.30 – 0.50 | Keep, pertimbangkan orthogonalisasi |
| < 0.30 | Keep both, independent signals |

### Final 6 Signal Set (setelah drop)

```
1. volatility
2. liquidity
3. volume_compression
4. reversal_1w
5. momentum_30d
6. taker_buy_contrarian
```

Semua pairwise correlation < 0.48 setelah drop → well-diversified factor set.

---

## Q5 — Composite Signal Construction

**Pertanyaan:** Bagaimana cara build composite signal yang robust?

### Arsitektur Dua Composite

| Composite | Signals | Rebalance | Holding |
|---|---|---|---|
| **Fast (h=1)** | volatility, liquidity, volume_compression, reversal_1w, taker_buy_contrarian | Daily | 1 hari |
| **Slow (h=20)** | volatility, liquidity, momentum_30d, price_to_high | Weekly | 10-20 hari |

Bisa run simultan dengan capital allocation terpisah (contoh: 60% fast / 40% slow).

### Formula Composite

**Step A — Z-score cross-sectional:**
$$z_{i,t}^s = \text{clip}\left(\frac{s_{i,t} - \mu_t^s}{\sigma_t^s},\ -3,\ 3\right)$$

**Step B — IC-weighted composite:**
$$C_{i,t} = \sum_{s=1}^{K} w^s \cdot z_{i,t}^s$$

**Step C — Re-normalize:**
$$\tilde{C}_{i,t} = \frac{C_{i,t} - \mu_t^C}{\sigma_t^C}$$

### Weights Fast Composite (h=1)

| Signal | IC (h=1) | Raw Weight | Normalized |
|---|---|---|---|
| volatility | +0.068 | +0.068 | +0.39 |
| liquidity | -0.042 | -0.042 | -0.24 |
| volume_compression | -0.026 | -0.026 | -0.15 |
| reversal_1w | +0.023 | +0.023 | +0.13 |
| taker_buy_contrarian | -0.013 | -0.013 | -0.08 |
| momentum_30d | +0.003 | — | **exclude** |

Normalized = raw / sum(|raw|). Sign embedded dalam weight.

### Momentum_30d — Exclude dari Fast Composite

- IC=0.003, t=0.35 di h=1 → noise
- Include → tambah turnover tanpa tambah return
- Corr=-0.478 dengan reversal → dilute reversal signal yang works

### Gram-Schmidt Orthogonalisasi — Tidak Perlu

Max correlation antar 5 sinyal fast composite hanya **0.28** → simple IC-weighting sudah cukup.

Rule of thumb: Orthogonalize hanya kalau N/K > 10 dan correlation > 0.5. Kita punya N/K = 26/5 = 5 → terlalu kecil, inject noise.

### Weight Estimation — Expanding Window (Recommended)

```python
def get_weights(ic_series_dict, min_days=120):
    weights = {}
    for signal, ic_ts in ic_series_dict.items():
        if len(ic_ts) < min_days:
            weights[signal] = 0.0
        else:
            ic_mean = ic_ts.mean()
            ic_se = ic_ts.std() / np.sqrt(len(ic_ts))
            if abs(ic_mean / ic_se) > 1.0:
                weights[signal] = ic_mean
            else:
                weights[signal] = 0.0
    return weights
```

**Key safeguard:** Jangan pernah flip sign dari prior expectation. Kalau volatility IC tiba-tiba negatif di expanding window → set weight = 0, bukan flip negatif.

### Expected Composite IC

$$IC_{composite} \approx 0.08 - 0.10$$

Lebih tinggi dari sinyal individual karena diversifikasi across orthogonal alphas.

---

## Q6 — Derivatives Data (Tardis.dev $19/bulan)

**Pertanyaan:** Worth it bayar Tardis untuk historical funding rate + OI?

### Verdict: ❌ Not Yet

| Factor | Assessment |
|---|---|
| Dollar cost | Trivial di AUM > $5k |
| Expected alpha uplift | Modest: +10-20% improvement composite IC |
| Implementation time | 2-3 hari download, parse, align, validate |
| Risk | Signal mungkin sudah degraded post-2022 |
| Alternative | 5-signal composite IC≈0.08-0.10 sudah kuat |

### Realistic Alpha dari Funding Rate

- Expected IC: 0.03-0.05 at h=1 (berdasarkan published research)
- Post-FTX: funding rate extremes lebih jarang dan di-arbitrage lebih cepat
- **Incremental IC ke composite: +0.005 sampai +0.015** — tidak transformative

### Break-even AUM

$$AUM_{break-even} = \frac{\$228/year}{58\%\ incremental\ return} = \$393$$

Secara dollar trivial di AUM > $400. Tapi bukan soal uang — soal **waktu implementasi**.

### Taker Buy vs Funding Rate

| Dimension | Funding Rate | Taker Buy Ratio |
|---|---|---|
| Yang diukur | Cost of holding longs vs shorts | Aggression market buyers vs sellers |
| Timeframe | 8-hour settlement, reflects positioning | Real-time, reflects flow |
| Overlap | Correlation 0.3-0.5 cross-sectionally | — |
| Unique info | Crowding in perpetual swaps | Directional aggression |

Partially substitute, tidak fully redundant. Tapi incremental contribution modest.

### Priority Sequence

```
1. Build & validate composite dari 5 sinyal existing
2. Run portfolio construction long/short
3. Measure realized Sharpe after costs
4. Kalau Sharpe < 1.5 → tambah funding/OI
5. Kalau Sharpe > 2.0 → marginal value data sangat rendah
```

**Cek Binance Vision `metrics/` folder dulu** — mungkin ada funding rate/OI gratis.

---

## Q7 — Portfolio Construction

**Pertanyaan:** Optimal portfolio construction untuk N=26 long-short strategy?

### Weighting Method: Signal-Proportional + Liquidity Cap

```python
def construct_portfolio(composite_z, liquidity_rank, max_weight=0.15):
    raw_w = composite_z.copy()
    liq_scale = liquidity_rank.clip(0.3, 1.0)
    raw_w *= liq_scale

    longs = raw_w[raw_w > 0]
    shorts = raw_w[raw_w < 0]

    longs = longs / longs.sum()
    shorts = shorts / shorts.abs().sum() * -1

    weights = pd.concat([longs, shorts])
    weights = weights.clip(-max_weight, max_weight)

    longs = weights[weights > 0]
    shorts = weights[weights < 0]
    longs = longs / longs.sum()
    shorts = shorts / shorts.abs().sum() * -1

    return pd.concat([longs, shorts])
```

Result: ~10-13 longs + ~10-13 shorts, weighted by conviction.

### Liquidity Tiering

| Tier | Volume 30d ADV | Treatment |
|---|---|---|
| Tier 1 | >$200M/day (BTC, ETH, SOL, BNB, XRP) | Full weight, cap 15% |
| Tier 2 | $50-200M/day (LINK, AVAX, ADA, DOT) | Full weight, cap 10% |
| Tier 3 | $10-50M/day (INJ, STX, SEI) | Scale 50%, cap 5% |
| Tier 4 | <$10M/day (RNDR thin days) | **Hard exclude** |

**Practical rule:** Exclude kalau target position > 0.5% dari symbol's 30d ADV.

### Leverage

| Metric | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Gross exposure | 2× (1L+1S) | 3× (1.5L+1.5S) | 4× (2L+2S) |
| Net exposure | 0 | 0 | 0 |
| Expected daily vol | ~2-3% | ~3-5% | ~4-7% |
| Min Sharpe needed | >1.0 | >1.5 | >2.0 |

**Recommendation:** Start 2× gross, scale ke 3× setelah 3 bulan live validation.

Kelly criterion: IC=0.08, daily Sharpe ≈ 0.23 → optimal leverage ~2-3×.

### Risk Management Framework

**Layer 1 — Individual Position:**
| Rule | Value |
|---|---|
| Max weight | 15% of leg |
| Individual stop-loss | **None** |
| Max notional per symbol | 0.5% of ADV |

> **Kenapa tidak ada individual stop-loss:** Dalam cross-sectional strategy, posisi individual bukan standalone bet. Short yang rugi mungkin posisi yang benar karena signal bilang harus underperform relatif terhadap yang lain. Stop individual destroy cross-sectional structure.

**Layer 2 — Portfolio Level:**
| Rule | Trigger | Action |
|---|---|---|
| Daily loss limit | -3% NAV/hari | Reduce gross 50% next day |
| Drawdown limit | -10% from peak | Reduce ke 1× sampai new high |
| Drawdown halt | -20% from peak | Flatten semua, review strategy |
| BTC beta spike | Beta > 0.3 (5d rolling) | Rebalance ke neutrality |
| Vol scaling | Realized vol > 2× target | Scale positions down |

**Layer 3 — Volatility Targeting (Paling Penting):**

```python
TARGET_VOL = 0.15   # 15% annualized
HALFLIFE = 20       # days EWMA

def vol_scale(daily_returns, target=TARGET_VOL):
    realized_vol = daily_returns.ewm(halflife=HALFLIFE).std() * np.sqrt(365)
    scale = target / realized_vol.iloc[-1]
    return min(scale, 2.0)  # never lever more than 2× base
```

Auto de-lever saat volatile regime, re-lever saat calm. Single most impactful risk technique.

### Minimum Viable Risk Framework Summary

```
Portfolio:
  Gross: 2× (1L + 1S), vol-targeted 15% annualized
  Net:   0 ± 5% (rebalance kalau |net| > 5%)

Positions:
  Max weight:  15% per leg
  Liq cap:     0.5% of ADV
  Hard exclude: <$10M ADV

Stops:
  No individual stops
  -3% daily  → halve exposure
  -10% DD    → reduce to 1×
  -20% DD    → flatten, review
```

---

## Master Action Items

### Immediate (Next Session)

- [ ] Update `ic_research.py`:
  - Drop `quote_liquidity`, `momentum_90d`, `price_to_high` dari signal set
  - Implement Newey-West t-stat (statsmodels, 5 lags)
  - Naikkan `MIN_SYMBOLS_PER_DATE = 15`
  - Tambah subsample stability check (2022-2023 vs 2024-2025)
  - Tambah 3 sinyal baru: CLV, taker imbalance momentum, intraday range ratio
  - Tambah median IC ke output

- [ ] Cek Binance Vision `metrics/` folder — ada funding rate/OI gratis?

### Next

- [ ] Buat `composite.py`:
  - Fast composite (5 sinyal, h=1, daily rebalance)
  - Slow composite (4 sinyal, h=20, weekly rebalance)
  - Expanding window weight estimation
  - Composite IC validation vs individual signals

- [ ] Buat `portfolio.py`:
  - Signal-proportional weighting + liquidity cap
  - Liquidity tiering (4 tiers)
  - Vol targeting 15% annualized (EWMA halflife=20)
  - Portfolio-level risk management rules

### Later (kalau Sharpe < 1.5)

- [ ] Evaluate Tardis.dev $19/month untuk funding rate + OI historical
- [ ] Implement funding_rate_contrarian + oi_price_signal
- [ ] Re-run IC analysis dengan full signal set

---

## Expected Performance

| Metric | Individual Signal Terbaik | Fast Composite |
|---|---|---|
| IC Mean | 0.068 (volatility) | ~0.08-0.10 |
| t-stat | 7.07 | >10 |
| Sharpe target | — | >1.5 |
| Leverage | — | 2× gross |
| Target vol | — | 15% annualized |

