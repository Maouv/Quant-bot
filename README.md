# Quant-Bot: Crypto Cross-Sectional Factor IC Analysis

Implementasi **Cross-Sectional Factor IC Analysis** untuk crypto futures, terinspirasi dari:
> Liu, Tsyvinski & Wu (2022) — *"Common Risk Factors in Cryptocurrency"* — Journal of Finance

---

## Model Matematika

### 1. Information Coefficient (IC) — Spearman Rank Correlation

Untuk setiap sinyal $s$ dan setiap hari $t$, IC dihitung secara **cross-sectional** across $N$ aset:

$$IC_t^s = \rho_{Spearman}\left(\text{rank}(s_{i,t}),\ \text{rank}(r_{i,t+h})\right)$$

di mana:
- $s_{i,t}$ = nilai sinyal aset $i$ pada hari $t$
- $r_{i,t+h}$ = forward return aset $i$ pada horizon $h$ hari ke depan
- $\rho_{Spearman}$ = Spearman rank correlation

$$IC_t = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}$$

di mana $d_i$ = perbedaan rank antara sinyal dan forward return untuk aset $i$.

---

### 2. Agregasi IC

$$\overline{IC} = \frac{1}{T} \sum_{t=1}^{T} IC_t$$

$$\sigma_{IC} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (IC_t - \overline{IC})^2}$$

$$ICIR = \frac{\overline{IC}}{\sigma_{IC}}$$

$$t\text{-stat} = \frac{\overline{IC}}{\sigma_{IC} / \sqrt{T}}$$

**Threshold signifikansi:**
- $|t\text{-stat}| > 2.0$ → statistically significant
- $ICIR > 0.5$ → sinyal kuat
- $\% IC > 0 > 55\%$ → sinyal konsisten

---

### 3. Preprocessing

**Z-score normalisasi** cross-sectional per hari (Huij & Verbeek, 2007):

$$z_{i,t}^s = \frac{s_{i,t} - \mu_t^s}{\sigma_t^s}, \quad z \in [-3, 3]$$

**Winsorisasi** forward return per hari (Fama-French methodology):

$$r_{i,t}^{winsor} = \text{clip}(r_{i,t},\ P_1,\ P_{99})$$

---

### 4. Factor Turnover

$$TO_t^s = \frac{1}{N} \sum_{i=1}^{N} \left| \text{rank}(s_{i,t}) - \text{rank}(s_{i,t-1}) \right| / N$$

$$\overline{TO}^s = \frac{1}{T} \sum_{t=1}^{T} TO_t^s$$

---

## Sinyal / Faktor

### 1. Reversal 1d

$$reversal\_1d_{i,t} = -\frac{close_{i,t} - close_{i,t-1}}{close_{i,t-1}}$$

**Fungsi:** Mean reversion jangka pendek. Menangkap overreaction pasar retail.

---

### 2. Reversal 1w

$$reversal\_1w_{i,t} = -\frac{close_{i,t} - close_{i,t-7}}{close_{i,t-7}}$$

**Fungsi:** Mean reversion mingguan. Pullback setelah weekly trend.

---

### 3. Momentum 30d

$$momentum\_30d_{i,t} = \frac{close_{i,t}}{close_{i,t-30}} - 1$$

**Fungsi:** Trend following jangka menengah. Bekerja optimal di horizon 10-20 hari.

---

### 4. Momentum 90d

$$momentum\_90d_{i,t} = \frac{close_{i,t}}{close_{i,t-90}} - 1$$

**Fungsi:** Trend following jangka panjang. Structural trend bukan noise.

---

### 5. Volatility

$$volatility_{i,t} = -\sigma_{30d}(r_{i,t}) = -\sqrt{\frac{1}{30}\sum_{k=0}^{29}(r_{i,t-k} - \bar{r})^2}$$

**Fungsi:** Low-volatility anomaly. Aset volatilitas rendah cenderung outperform (risk-adjusted).

---

### 6. Liquidity

$$liquidity_{i,t} = \frac{1}{30}\sum_{k=0}^{29}\frac{volume_{i,t-k}}{high_{i,t-k} - low_{i,t-k}}$$

**Fungsi:** Amihud liquidity proxy. Negatif IC → aset likuid justru underperform jangka pendek.

---

### 7. Vol Compression

$$vol\_compression_{i,t} = \frac{\sigma_{5d}(r_{i,t})}{\sigma_{30d}(r_{i,t})}$$

**Fungsi:** Deteksi volatilitas menyempit. Ratio < 1 → potensi breakout.

---

### 8. Volume Compression

$$volume\_compression_{i,t} = \frac{volume_{i,t}}{\frac{1}{30}\sum_{k=0}^{29}volume_{i,t-k}}$$

**Fungsi:** Volume anomaly. Volume spike → aktivitas unusual → cenderung underperform besok.

---

### 9. Price to High

$$price\_to\_high_{i,t} = \frac{close_{i,t}}{\max_{k=0}^{29}(high_{i,t-k})}$$

**Fungsi:** Proximity ke 30d high. Nilai mendekati 1 → momentum kuat.

---

### 10. Taker Buy Contrarian

$$taker\_buy\_ratio_{i,t} = \frac{1}{3}\sum_{k=0}^{2}\frac{taker\_buy\_volume_{i,t-k}}{volume_{i,t-k}}$$

$$taker\_buy\_contrarian_{i,t} = -taker\_buy\_ratio_{i,t}$$

**Fungsi:** Retail sentiment reversal. Retail agresif buy → contrarian signal.

---

### 11. Quote Liquidity

$$quote\_liquidity_{i,t} = \frac{1}{30}\sum_{k=0}^{29}\frac{quote\_volume_{i,t-k}}{high_{i,t-k} - low_{i,t-k}}$$

**Fungsi:** Liquidity dalam USD terms. Lebih akurat untuk cross-symbol comparison.

---

## Universe & Data

| Item | Detail |
|---|---|
| **Exchange** | Binance Futures (USD-Margined) |
| **Source** | Binance Vision `data/futures/um/daily/klines/` |
| **Interval** | 1 hari (daily) |
| **Range** | 2022-01-01 s/d 2025-12-31 |
| **Total Symbols** | 26 |

### Daftar Symbols

| Symbol | Rows | Keterangan |
|---|---|---|
| BTCUSDT | 1096 | Resample dari 1h |
| ETHUSDT | 1096 | Resample dari 1h |
| BNBUSDT | 1096 | Resample dari 1h |
| SOLUSDT | 1096 | Resample dari 1h |
| XRPUSDT | 1456 | Download ulang Binance Vision |
| SUIUSDT | 974 | Listing 2023 |
| ADAUSDT | 1461 | Full 4 tahun |
| AVAXUSDT | 1461 | Full 4 tahun |
| DOTUSDT | 1461 | Full 4 tahun |
| LINKUSDT | 1461 | Full 4 tahun |
| LTCUSDT | 1455 | Full 4 tahun |
| ATOMUSDT | 1461 | Full 4 tahun |
| NEARUSDT | 1455 | Full 4 tahun |
| UNIUSDT | 1461 | Full 4 tahun |
| MKRUSDT | 1455 | Full 4 tahun |
| AAVEUSDT | 1461 | Full 4 tahun |
| APTUSDT | 1170 | Listing 2022-10 |
| ARBUSDT | 1015 | Listing 2023-03 |
| OPUSDT | 1310 | Listing 2022-06 |
| INJUSDT | 1233 | Listing 2022-08 |
| FETUSDT | 1080 | Listing 2022-10 |
| STXUSDT | 1045 | Listing 2023-01 |
| TIAUSDT | 793 | Listing 2023-09 |
| SEIUSDT | 868 | Listing 2023-07 |
| WLDUSDT | 892 | Listing 2023-06 |
| RNDRUSDT | 544 | Data terbatas |

---

## Hasil Test — Run Pertama

**Config:**
- In-sample: 70% (1022 hari, 2022-2024)
- Out-of-sample: 30% (439 hari, 2024-2025)
- Min symbols per date: 5
- Horizon: 1d

### IC Mean at Horizon = 1

| Sinyal | IC Mean | t-stat | ICIR | Status |
|---|---|---|---|---|
| volatility | +0.068 | **7.07** | 0.187 | ✅ Significant |
| liquidity | -0.042 | **-5.66** | -0.150 | ✅ Significant |
| quote_liquidity | -0.039 | **-5.42** | -0.143 | ✅ Significant |
| volume_compression | -0.026 | **-3.86** | -0.102 | ✅ Significant |
| reversal_1w | +0.023 | **3.01** | 0.079 | ✅ Significant |
| reversal_1d | +0.018 | **2.33** | 0.061 | ✅ Significant |
| price_to_high | +0.016 | 1.90 | 0.050 | ⚠️ Borderline |
| taker_buy_contrarian | -0.013 | -1.84 | -0.048 | ⚠️ Borderline |
| momentum_30d | +0.003 | 0.35 | 0.009 | ❌ Noise di h=1 |
| momentum_90d | -0.002 | -0.23 | -0.006 | ❌ Noise di h=1 |
| vol_compression | -0.001 | -0.10 | -0.003 | ❌ Noise di h=1 |

### Top 3 by ICIR (semua horizon)

| Sinyal | ICIR | t-stat | Horizon |
|---|---|---|---|
| volatility | 0.327 | 12.29 | 20d |
| volatility | 0.290 | 10.91 | 10d |
| price_to_high | 0.272 | 10.21 | 20d |

### Key Insights dari IC Decay

| Sinyal | Insight |
|---|---|
| **volatility** | IC naik terus s/d horizon 20d → optimal untuk swing/position trading |
| **liquidity** | IC makin negatif → efek makin kuat di holding period panjang |
| **momentum_30d** | IC rendah di h=1 tapi naik signifikan di h=10-20 → butuh holding lebih lama |
| **reversal_1d/1w** | IC positif di horizon pendek, membalik negatif di h=20 → murni mean reversion |
| **taker_buy_contrarian** | IC makin negatif seiring horizon → retail buying = bearish signal yang persist |

### In-Sample vs Out-of-Sample

| Sinyal | IC In-Sample | IC Out-of-Sample | Degradasi |
|---|---|---|---|
| volatility | 0.061 | 0.084 | ✅ Lebih baik di OOS |
| liquidity | -0.032 | -0.065 | ✅ Lebih kuat di OOS |
| reversal_1w | 0.022 | 0.027 | ✅ Stabil |
| momentum_30d | -0.004 | +0.018 | ✅ OOS lebih baik |

Tidak ada sinyal yang degradasi signifikan di OOS → tidak ada overfitting.

---

## Progress

### ✅ Selesai
- Framework IC analysis dipilih (Liu et al. crypto factor model)
- 11 sinyal didefinisikan + formula matematis lengkap
- Data pipeline: download Binance Vision + merge per symbol
- 26 symbols siap, range 2022-2025
- Script `ic_research.py` single file, jalan di VPS
- Run pertama berhasil: `ic_results.csv`, `ic_chart.png`, `ic_decay_chart.png`, `ic_summary.txt`
- Identified sinyal significant (t > 2.0): volatility, liquidity, quote_liquidity, volume_compression, reversal_1w, reversal_1d
- Identified sinyal noise di h=1: momentum_30d, momentum_90d, vol_compression

### ⚠️ Setengah Jalan
- Tweak parameter `PRICE_HIGH_WIN=14`, `TAKER_WIN=7`, `VOL_SHORT=10` — sudah direncanakan, belum dirun
- Tambah sinyal `reversal_2w` — sudah direncanakan, belum diimplementasi
- Eksplorasi `data/futures/um/daily/metrics/` di Binance Vision — folder ditemukan, belum dicek kolom

### ❌ Belum Dimulai
- Composite signal dari sinyal significant
- In/out-of-sample validation review formal
- Portfolio construction (long top quintile, short bottom quintile)
- Parameter sensitivity analysis formal
- Integrasi data metrics (OI, funding rate) jika tersedia di Binance Vision

---

## Next Steps Priority

1. **Run tweak parameter** → cek apakah `price_to_high` dan `taker_buy_contrarian` t-stat naik
2. **Cek `metrics/` folder** → kalau ada OI + funding rate, tambah sinyal derivatives
3. **Composite signal** → gabungkan sinyal significant jadi satu score
4. **Portfolio construction** → long top quintile, short bottom quintile

---

## File Structure

```
quant-bot/
├── ic_research.py        # Script utama IC analysis
├── merged_data/          # Data CSV per symbol (1d)
│   ├── BTCUSDT-1d-full.csv
│   ├── ETHUSDT-1d-full.csv
│   └── ... (26 symbols)
├── ic_results.csv        # Output: IC per sinyal per horizon
├── ic_chart.png          # Output: Bar chart IC Mean h=1
├── ic_decay_chart.png    # Output: IC decay by horizon
└── ic_summary.txt        # Output: Text summary
```

---

## Requirements

```
pandas
numpy
scipy
matplotlib
```

```bash
pip install pandas numpy scipy matplotlib
python3 ic_research.py
```

