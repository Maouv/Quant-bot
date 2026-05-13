# Metrics Data Implementation Plan
> Binance Vision `data/futures/um/daily/metrics/` — Free Historical Derivatives Data

---

## Discovery

File: `AVAXUSDT-metrics-2026-05-12.zip`  
Source: `https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/`

### Kolom yang Tersedia (interval 5 menit, 288 rows/hari)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `create_time` | timestamp | 5-minute intervals |
| `symbol` | str | Symbol name |
| `sum_open_interest` | float | OI dalam coin units |
| `sum_open_interest_value` | float | OI dalam USD |
| `count_toptrader_long_short_ratio` | float | Top trader L/S ratio (count-based) |
| `sum_toptrader_long_short_ratio` | float | Top trader L/S ratio (volume-based) |
| `count_long_short_ratio` | float | All users L/S ratio |
| `sum_taker_long_short_vol_ratio` | float | Taker long vs short volume ratio |

---

## Opus Assessment — Expected IC Uplift

| Signal | Expected IC (h=1) | Corr dengan Existing | Incremental IC ke Composite |
|---|---|---|---|
| OI-price divergence | 0.02-0.04 | Low (~0.1-0.2 vs volatility) | **+0.01-0.02** |
| L/S ratio contrarian (all users) | 0.03-0.05 | Moderate (~0.4-0.5 vs taker_buy) | **+0.005-0.015** |
| Top trader L/S contrarian | 0.02-0.04 | Moderate (~0.3-0.4 vs L/S) | **+0.005-0.01** |
| Taker L/S vol ratio | 0.02-0.03 | **High (~0.85-0.95 vs taker_buy_contrarian)** | ~0 → **SKIP** |

**Total realistic uplift:** +0.02-0.04 → composite IC dari ~0.08-0.10 ke **~0.10-0.14** (20-40% improvement)

---

## Signal Formulas

### 1. OI-Price Divergence (Priority #1)

$$oi\_price\_signal_{i,t} = -\left(\frac{\Delta OI_{i,t}^{5d}}{OI_{i,t-5}} - \frac{\Delta price_{i,t}^{5d}}{price_{i,t-5}}\right)$$

**Logika:**
- OI naik + price turun → new shorts opening → crowded short → squeeze signal
- OI naik + price naik → new longs → crowded long → vulnerable to reversal
- **Genuinely orthogonal** ke semua OHLCV signals

```python
# Daily aggregation: LAST value (OI adalah stock variable)
oi_value = df_5min['sum_open_interest_value'].iloc[-1]
oi_change_pct = (df_5min['sum_open_interest_value'].iloc[-1] /
                 df_5min['sum_open_interest_value'].iloc[0]) - 1

# Signal computation (setelah merge dengan OHLCV)
oi_pct_5d = oi_value / oi_value.shift(5) - 1
price_pct_5d = close / close.shift(5) - 1
oi_price_signal = -(oi_pct_5d - price_pct_5d)
```

---

### 2. L/S Ratio Contrarian — All Users (Priority #2)

$$ls\_contrarian_{i,t} = -\overline{count\_long\_short\_ratio}_{i,t}$$

**Logika:** Retail crowd positioning. Ketika semua orang long (ratio >> 1) → contrarian short signal. Ini adalah **"funding rate proxy" yang sebelumnya mau dibayar $19/bulan** — ternyata gratis di sini.

```python
# Daily aggregation: MEAN (ratio fluctuates intraday)
ls_ratio_mean = df_5min['count_long_short_ratio'].mean()

# Signal
ls_contrarian = -ls_ratio_mean
```

---

### 3. Top Trader L/S Contrarian (Priority #3)

$$top\_trader\_contrarian_{i,t} = -\overline{sum\_toptrader\_long\_short\_ratio}_{i,t}$$

**Logika:** Top traders lebih smart dari retail — contrarian signal lebih lemah. **Perlu ditest apakah IC positif (fade top traders) atau negatif (follow top traders).**

```python
# Daily aggregation: MEAN
top_trader_ls_mean = df_5min['sum_toptrader_long_short_ratio'].mean()

# Signal (test dua arah dulu)
top_trader_contrarian = -top_trader_ls_mean  # contrarian
# top_trader_follow = +top_trader_ls_mean    # follow — test juga
```

---

### 4. Taker L/S Vol Ratio — SKIP

Redundant dengan `taker_buy_contrarian` yang sudah ada.

| | `taker_buy_contrarian` (OHLCV) | `sum_taker_long_short_vol_ratio` (metrics) |
|---|---|---|
| Formula | `taker_buy_vol / total_vol` | `taker_long_vol / taker_short_vol` |
| Relationship | X/Y | X/(Y-X) → monotonic transform |
| Expected rank corr | — | **0.85-0.95** |

→ Zero incremental information. Skip.

---

## Aggregation Rules per Kolom

| Kolom | Aggregation Method | Alasan |
|---|---|---|
| `sum_open_interest_value` | **Last value** (23:55 UTC) | Stock variable, end-of-day snapshot |
| `sum_open_interest` | **Last value** | Same |
| `count_long_short_ratio` | **Daily mean** | Ratio fluctuates, mean captures day's bias |
| `sum_toptrader_long_short_ratio` | **Daily mean** | Same logic |
| `sum_taker_long_short_vol_ratio` | **Skip** | Redundant |

**Bonus aggregations untuk OI:**
```python
def aggregate_metrics_daily(df_5min):
    return {
        'oi_value':           df_5min['sum_open_interest_value'].iloc[-1],
        'oi_change_pct':      (df_5min['sum_open_interest_value'].iloc[-1] /
                               df_5min['sum_open_interest_value'].iloc[0]) - 1,
        'ls_ratio_mean':      df_5min['count_long_short_ratio'].mean(),
        'top_trader_ls_mean': df_5min['sum_toptrader_long_short_ratio'].mean(),
    }
```

---

## Implementation Plan (Staged Approach)

### Phase 1 — Validation Sample (1-2 jam)

Download 3 bulan data untuk 10 liquid symbols dulu:

```python
SYMBOLS_SAMPLE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT"
]
# ~900 files, manageable
# Periode: 2024-01 sampai 2024-03
```

**Goal Phase 1:**
- Validasi data quality dan completeness
- Cek actual date range yang tersedia
- Hitung IC sinyal baru sebelum commit ke full download
- Konfirmasi OI-price IC > 0.02 dan L/S IC > 0.03

---

### Phase 2 — Full Download (4 jam unattended)

Kalau Phase 1 confirm IC, jalankan full download overnight.

**Estimasi:**
- 26 symbols × ~3 years × 365 days = ~28,000 zip files
- ~5KB per file → ~140MB total
- Rate: 2 requests/second → ~4 jam

```python
import time
import urllib.request
import os
from pathlib import Path
from datetime import date, timedelta

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"

def download_metrics(symbol, date_str, out_dir):
    fname = f"{symbol}-metrics-{date_str}.zip"
    url = f"{BASE}/{symbol}/{fname}"
    out_path = Path(out_dir) / symbol / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return "skip"  # already downloaded
    try:
        urllib.request.urlretrieve(url, out_path)
        return "ok"
    except Exception as e:
        return f"miss"  # missing days are normal
    finally:
        time.sleep(0.5)  # rate limit — be polite

# Run
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "FETUSDT",
    "STXUSDT", "TIAUSDT", "SEIUSDT", "WLDUSDT", "RNDRUSDT",
    "ATOMUSDT", "DOTUSDT", "LTCUSDT", "MKRUSDT", "UNIUSDT", "AAVEUSDT"
]

start = date(2022, 1, 1)
end = date(2025, 12, 31)
OUT_DIR = "./metrics_raw"

d = start
while d <= end:
    for sym in SYMBOLS:
        result = download_metrics(sym, d.strftime("%Y-%m-%d"), OUT_DIR)
        if result == "ok":
            print(f"✅ {sym} {d}")
        elif result == "miss":
            pass  # silent skip
    d += timedelta(days=1)
```

---

### Phase 3 — Merge ke Daily CSV

Setelah download, aggregate semua 5-min files ke satu row per hari per symbol.

```python
import zipfile
import pandas as pd
import glob
from pathlib import Path

def process_symbol_metrics(symbol, raw_dir, out_dir):
    """Merge semua daily metrics files untuk satu symbol."""
    files = sorted(glob.glob(f"{raw_dir}/{symbol}/{symbol}-metrics-*.zip"))
    daily_rows = []

    for fp in files:
        try:
            with zipfile.ZipFile(fp) as z:
                df = pd.read_csv(z.open(z.namelist()[0]))

            if len(df) < 50:  # partial day — skip
                continue

            row = aggregate_metrics_daily(df)
            # Extract date from filename
            date_str = Path(fp).stem.replace(f"{symbol}-metrics-", "")
            row['date'] = pd.to_datetime(date_str)
            daily_rows.append(row)
        except Exception:
            continue

    if not daily_rows:
        print(f"⚠️ No data for {symbol}")
        return

    result = pd.DataFrame(daily_rows).set_index('date').sort_index()
    out_path = f"{out_dir}/{symbol}-metrics-daily.csv"
    result.to_csv(out_path)
    print(f"✅ {symbol}: {len(result)} days → {out_path}")
```

**Output:** `merged_metrics/BTCUSDT-metrics-daily.csv` per symbol, dengan kolom:
```
date, oi_value, oi_change_pct, ls_ratio_mean, top_trader_ls_mean
```

---

### Phase 4 — Integrasi ke `ic_research.py`

Tambah 3 sinyal baru ke `compute_signals()`:

```python
# Load metrics data
metrics_dict = {}
for fp in glob.glob('./merged_metrics/*-metrics-daily.csv'):
    sym = os.path.basename(fp).replace('-metrics-daily.csv', '')
    metrics_dict[sym] = pd.read_csv(fp, index_col='date', parse_dates=True)

# Dalam compute_signals(sym):
if sym in metrics_dict:
    m = metrics_dict[sym].reindex(df.index)

    # OI price signal
    oi_pct_5d = m['oi_value'].pct_change(5)
    price_pct_5d = df['close'].pct_change(5)
    df['oi_price_signal'] = -(oi_pct_5d - price_pct_5d)

    # L/S ratio contrarian
    df['ls_ratio_contrarian'] = -m['ls_ratio_mean']

    # Top trader contrarian
    df['top_trader_contrarian'] = -m['top_trader_ls_mean']
```

---

## Handling Missing Data

| Kasus | Treatment |
|---|---|
| Missing day (file tidak ada) | Normal — skip, NaN di merged CSV |
| Partial day (<50 rows dari 288) | Treat as missing |
| Symbol dengan <6 bulan metrics data | Exclude dari metrics signals, tetap di OHLCV signals |
| Gaps di tengah series | Forward-fill OI level (tidak berubah kalau tidak ada file) |
| L/S ratio gaps | NaN — biarkan IC calculation handle via `min_symbols` filter |

```python
# Forward-fill OI only (stock variable)
merged['oi_value'] = merged['oi_value'].ffill()
merged['oi_change_pct'] = merged['oi_change_pct']  # jangan ffill ratio change

# L/S ratios — biarkan NaN
merged['ls_ratio_mean'] = merged['ls_ratio_mean']  # no ffill
```

---

## Storage Estimate

| Item | Size |
|---|---|
| 28,000 zip files (raw) | ~140MB |
| Merged daily CSVs (26 symbols) | <5MB |
| Total | **~145MB** — trivial |

---

## Decision Gate — Go/No-Go Full Download

Setelah Phase 1 (validation sample), cek:

| Metric | Threshold | Action kalau tidak terpenuhi |
|---|---|---|
| OI-price signal IC (h=1) | > 0.02 | Drop signal, skip full download |
| L/S ratio IC (h=1) | > 0.02 | Drop signal |
| Data availability | >80% hari ada untuk liquid symbols | Proceed anyway |
| Top trader IC direction | IC positif atau negatif konsisten | Keep signal apapun arahnya |

Kalau **kedua signal IC < 0.02** setelah Phase 1 → skip full download, fokus ke composite dari OHLCV saja.

---

## Updated Signal Set (Post-Metrics Integration)

### Fast Composite (h=1) — Updated

| Signal | Source | Expected IC |
|---|---|---|
| volatility | OHLCV | +0.068 |
| liquidity | OHLCV | -0.042 |
| volume_compression | OHLCV | -0.026 |
| reversal_1w | OHLCV | +0.023 |
| taker_buy_contrarian | OHLCV | -0.013 |
| **oi_price_signal** | **metrics** | **+0.02-0.04** |
| **ls_ratio_contrarian** | **metrics** | **+0.03-0.05** |
| **top_trader_contrarian** | **metrics** | **+0.02-0.04** |

**Expected composite IC setelah integrasi: ~0.10-0.14**

---

## Progress Checklist

- [ ] **Phase 1:** Download sample 10 symbols × 3 bulan
- [ ] **Phase 1:** Aggregate ke daily, hitung IC 3 sinyal baru
- [ ] **Phase 1:** Decision gate — IC confirmed?
- [ ] **Phase 2:** Full download 26 symbols × 3 tahun (overnight)
- [ ] **Phase 3:** Merge semua ke `merged_metrics/` CSV
- [ ] **Phase 4:** Integrasi ke `ic_research.py`
- [ ] **Phase 4:** Re-run IC analysis dengan sinyal baru
- [ ] **Phase 4:** Update `composite.py` dengan weights baru

