# SuperSignal Development Guide

## Safe Branch Workflow

All performance work for this pass is isolated on:

```text
feature/performance-optimization-v2
```

Do not deploy this branch directly to the production Streamlit app. Validate locally or in a temporary Streamlit Cloud deployment before merging into `main`.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Recommended validation flow:

1. Start the app locally.
2. Open the Market Overview and Technical Analysis tabs first.
3. Change symbol and timeframe repeatedly.
4. Confirm Smart Money, Order Book, MTF, AI Signals, Backtesting, and Portfolio tabs render without stale data.
5. Compare signal values against the current production app for the same symbol/timeframe.

## Architecture Notes

The application is organized around a Streamlit entrypoint (`app.py`) and focused modules under `src/`:

- `src/data`: market data, order book, CoinGecko, sentiment, Fear and Greed
- `src/analysis`: indicators, advanced indicators, SMC, MTF, support/resistance, backtesting, signal rules
- `src/ml`: feature preparation and optional model training
- `src/risk`: risk and position sizing utilities
- `src/ui`: layout, chart, sidebar, and signal card helpers

## Performance Practices

- Use `st.cache_data` for deterministic, serializable results.
- Use `st.cache_resource` or process-level caching for reusable clients and heavyweight resources.
- Keep exchange/API calls out of Streamlit render code where possible.
- Prefer batched provider calls over per-symbol loops.
- Avoid loading ML libraries at startup unless the current view needs them.
- Key `st.session_state` data by symbol/timeframe to avoid stale tab data.

## Rollback

Because production remains on `main`, rollback is simply declining or reverting the pull request branch. No production deployment should point at this branch until review and benchmark approval are complete.
