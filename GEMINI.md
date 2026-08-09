# StatArb Project Instructions

## Context
- **Core**: Pairs trading via cointegration.
- **Math**: Kalman Filter (dynamic $\beta$), ADF test (stationarity), OU model (mean reversion).
- **Split**: C++ (performance math), Python (HMM regime, visuals, tearsheets).

## Structure
- `cpp/src/`: Performance engine. No Eigen/heavy libs; uses flat arrays for 2x2.
- `python/`: Analytics and visualisation.
- `data/prices/`: Input OHLCV CSVs.

## Commands
- Build: `cd cpp; cmake -B build; cmake --build build`
- WF Backtest: `./build/statarb --mode wf --tickers T1,T2`
- Tests: `./build/test_statarb`

## Logic
- Entry: $|z| > 2.0$.
- Exit: $|z| < 0.5$.
- Stop: $|z| > 3.5$.
