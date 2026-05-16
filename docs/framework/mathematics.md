# Mathematical Framework

## Information Coefficient (IC)

Cross-sectional Spearman rank correlation between signal and forward returns:

$$IC_t^s = \rho_{Spearman}\left(\text{rank}(s_{i,t}),\ \text{rank}(r_{i,t+h})\right)$$

Where:
- $s_{i,t}$ = signal value for asset $i$ on day $t$
- $r_{i,t+h}$ = forward return for asset $i$ over horizon $h$
- $\rho_{Spearman}$ = Spearman rank correlation

## IC Aggregation

$$\overline{IC} = \frac{1}{T} \sum_{t=1}^{T} IC_t$$

$$\sigma_{IC} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (IC_t - \overline{IC})^2}$$

$$ICIR = \frac{\overline{IC}}{\sigma_{IC}}$$

$$t\text{-stat} = \frac{\overline{IC}}{\sigma_{IC} / \sqrt{T}}$$

## Newey-West t-statistic

Corrects for autocorrelation in IC series (5 lags):

$$\text{Var}_{NW} = \gamma_0 + 2\sum_{l=1}^{5} \left(1 - \frac{l}{6}\right) \gamma_l$$

$$t_{NW} = \frac{\overline{IC}}{\sqrt{\text{Var}_{NW} / T}}$$

Where $\gamma_l$ = autocovariance at lag $l$.

## Preprocessing

**Z-score normalization (cross-sectional per day):**

$$z_{i,t}^s = \frac{s_{i,t} - \mu_t^s}{\sigma_t^s}, \quad z \in [-3, 3]$$

**Forward return winsorization (per day):**

$$r_{i,t}^{winsor} = \text{clip}(r_{i,t},\ P_1,\ P_{99})$$

## Signal Formulas

| Signal | Formula |
|---|---|
| reversal_1d | $-\frac{close_t - close_{t-1}}{close_{t-1}}$ |
| reversal_1w | $-\frac{close_t - close_{t-7}}{close_{t-7}}$ |
| momentum_30d | $\frac{close_t}{close_{t-30}} - 1$ |
| volatility | $-\sigma_{30d}(r_t)$ |
| liquidity | $\frac{1}{30}\sum_{k=0}^{29}\frac{volume_{t-k}}{high_{t-k} - low_{t-k}}$ |
| vol_compression | $\frac{\sigma_{5d}(r_t)}{\sigma_{30d}(r_t)}$ |
| volume_compression | $\frac{volume_t}{\frac{1}{30}\sum_{k=0}^{29}volume_{t-k}}$ |
| price_to_high | $\frac{close_t}{\max_{k=0}^{29}(high_{t-k})}$ |
| taker_buy_contrarian | $-\frac{1}{3}\sum_{k=0}^{2}\frac{taker\_buy\_volume_{t-k}}{volume_{t-k}}$ |

**Status:** Production
