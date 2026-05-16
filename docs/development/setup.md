# Development Setup

## Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scipy matplotlib statsmodels

# Verify
python3 -c "import pandas, numpy, scipy, matplotlib, statsmodels; print('OK')"
```

## Running the Pipeline

```bash
# 1. IC Research (discover signals)
python3 ic_research.py
# Output: ic_results.csv, ic_chart.png, ic_decay_chart.png, ic_summary.txt

# 2. Composite Signals (build composites)
python3 composite.py
# Output: composite/composite_scores.csv, composite_ic.csv

# 3. Portfolio Simulation (simulate P&L)
python3 portofolio.py
# Output: portfolio/portfolio_pnl.csv, portfolio_metrics.txt
```

## Testing

```bash
# Check data loading
python3 -c "
import glob
import pandas as pd
files = glob.glob('merged_data/*.csv')
print(f'Found {len(files)} OHLCV files')
for f in files[:3]:
    df = pd.read_csv(f)
    print(f'{f}: {len(df)} rows')
"

# Check signal computation
python3 -c "
from ic_research import load_data, compute_signals
symbols = load_data()
signals = compute_signals(symbols)
print(f'Computed {len(signals)} signals')
for sig, df in list(signals.items())[:3]:
    print(f'{sig}: {df.shape}')
"
```

## Code Style

- **Naming:** snake_case for functions/variables, UPPER_CASE for constants
- **Comments:** Explain why, not what
- **Functions:** Keep <100 lines, single responsibility
- **Docstrings:** Brief one-liner for each function

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-signal

# Make changes, test
python3 ic_research.py

# Commit
git add .
git commit -m "Add new signal: xyz"

# Push
git push origin feature/new-signal
```

## Debugging

```bash
# Enable verbose output
python3 ic_research.py 2>&1 | tee debug.log

# Check intermediate outputs
python3 -c "
import pandas as pd
ic_df = pd.read_csv('ic_results.csv')
print(ic_df[ic_df['horizon']==1].sort_values('t_stat', ascending=False))
"

# Profile performance
python3 -m cProfile -s cumtime ic_research.py | head -20
```

**Status:** Production
