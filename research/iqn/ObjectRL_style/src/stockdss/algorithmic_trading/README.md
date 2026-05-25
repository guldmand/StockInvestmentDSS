# Updated StockDSS algorithmic trading files

Copy the `algorithmic_trading` folder into:

```text
src/stockdss/algorithmic_trading
```

This version changes algorithmic trading outputs from:

```text
algorithmic_trading/files/<strategy>/
```

to:

```text
algorithmic_trading/results/<strategy>/<run_name>/
algorithmic_trading/plots/<strategy>/<run_name>/
algorithmic_trading/summary/
```

## Smoke test

```powershell
$env:PYTHONPATH="src"

python -m stockdss.algorithmic_trading.experiments.run_all_algorithmic_experiments `
  --trade-data data/trade_data_pit_500_2020_01_01_2025_12_31.csv `
  --dataset-tag pit_500_2020_01_01_2025_12_31 `
  --ticker AAPL `
  --run-root outputs/runs/test_algorithmic_trading_2020_2025_grid `
  --initial-amount 1000000 `
  --sma-grid "20:50,50:200,100:300" `
  --ema-grid "12:26,20:50" `
  --momentum-windows "20,60,120" `
  --breakout-windows "20,60" `
  --volatility-grid "20:20:0.4,60:20:0.4"
```
