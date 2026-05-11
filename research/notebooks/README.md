# Research Notebooks Index

This folder contains the research notebooks used for the StockInvestmentDSS thesis PoC.

The notebooks belong to the research/slow-layer track. Their outputs may later be used by the DSS application as demo evidence, cached metrics, model metadata, figures or decision-support examples.

## Runtime assumptions

Default local research DuckDB path:

```txt
../system/runtime-data/market_research.duckdb
```

Canonical NAS DuckDB path when mounted:

```txt
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

Expected output folders:

```txt
research/results/
research/results/tables/
research/results/figures/
research/results/models/
research/results/logs/
```

## Notebook overview

| Notebook | Mandatory V1.0 | Layer | Purpose |
|---|---:|---|---|
| `00_data_check.ipynb` | Yes | Slow / shared | Verify data access, DuckDB path, basic market data and reproducible output paths. |
| `01_finrl_baseline.ipynb` | Yes | Slow | Run or document the first minimal FinRL baseline flow. |
| `02_gymnasium_env.ipynb` | Yes | Slow | Verify the custom or wrapped trading environment interface. |
| `03_baseline_comparison.ipynb` | Yes | Slow | Compare simple non-RL baselines against RL/FinRL outputs. |
| `04_iqn_experiment.ipynb` | No | Slow | Prototype IQN/distributional RL experiment if time allows. |
| `05_uncertainty_proxy.ipynb` | No | Slow | Prototype uncertainty proxy or placeholder uncertainty metrics. |
| `06_thesis_figures.ipynb` | Yes | Slow / report | Generate thesis-ready tables and figures from verified results. |

---

## `00_data_check.ipynb`

Purpose:

Verify that the research environment can access data paths, DuckDB and minimal market data inputs.

Inputs:

```txt
research/.env
../system/runtime-data/market_research.duckdb
data/
```

Outputs:

```txt
research/results/tables/data_check_summary.csv
research/results/figures/data_check_overview.png
```

Required dependencies:

```txt
duckdb
pandas
numpy
yfinance
python-dotenv
```

Mandatory for V1.0:

```txt
Yes
```

Relation to thesis:

Supports the data section, reproducibility section and PoC validation.

Relation to application track:

Confirms that the shared analytical data path can be used later by backend/demo components.

---

## `01_finrl_baseline.ipynb`

Purpose:

Verify a minimal FinRL baseline workflow and document whether FinRL can be used as the research foundation.

Inputs:

```txt
research/.env
../system/runtime-data/market_research.duckdb
research/results/
```

Outputs:

```txt
research/results/tables/finrl_baseline_metrics.csv
research/results/logs/finrl_baseline_log.txt
```

Required dependencies:

```txt
finrl
torch
stable-baselines3
gymnasium
pandas
numpy
```

Mandatory for V1.0:

```txt
Yes
```

Relation to thesis:

Supports baseline experiment description, method validation and empirical results.

Relation to application track:

May produce stored model metrics or demo evidence for the DSS interface.

---

## `02_gymnasium_env.ipynb`

Purpose:

Verify that the trading environment follows the Gymnasium-style reset/step interface.

Inputs:

```txt
research/.env
market data from DuckDB or prepared Parquet/CSV
```

Outputs:

```txt
research/results/tables/gymnasium_env_check.csv
research/results/logs/gymnasium_env_check.txt
```

Required dependencies:

```txt
gymnasium
pandas
numpy
```

Mandatory for V1.0:

```txt
Yes
```

Relation to thesis:

Supports the environment design and RL formulation sections.

Relation to application track:

Provides confidence that later decision logic can rely on a consistent environment definition.

---

## `03_baseline_comparison.ipynb`

Purpose:

Compare simple baseline strategies with RL/FinRL outputs.

Inputs:

```txt
research/results/tables/finrl_baseline_metrics.csv
market data from DuckDB or prepared Parquet/CSV
```

Outputs:

```txt
research/results/tables/baseline_comparison.csv
research/results/figures/baseline_comparison.png
```

Required dependencies:

```txt
pandas
numpy
matplotlib
duckdb
```

Mandatory for V1.0:

```txt
Yes
```

Relation to thesis:

Supports the evaluation and results sections.

Relation to application track:

Can provide simplified demo metrics and comparison evidence.

---

## `04_iqn_experiment.ipynb`

Purpose:

Prototype the IQN/distributional RL idea if time allows.

Inputs:

```txt
prepared environment outputs
market data from DuckDB or Parquet/CSV
```

Outputs:

```txt
research/results/tables/iqn_experiment_metrics.csv
research/results/models/
research/results/logs/iqn_experiment_log.txt
```

Required dependencies:

```txt
torch
gymnasium
pandas
numpy
```

Mandatory for V1.0:

```txt
No
```

Relation to thesis:

Supports the advanced method section if implemented.

Relation to application track:

Not required for the first PoC demo.

---

## `05_uncertainty_proxy.ipynb`

Purpose:

Prototype uncertainty-related output for decision support if full evidential uncertainty is not ready.

Inputs:

```txt
model metrics
baseline comparison outputs
decision examples
```

Outputs:

```txt
research/results/tables/uncertainty_proxy.csv
research/results/figures/uncertainty_proxy.png
```

Required dependencies:

```txt
pandas
numpy
matplotlib
```

Mandatory for V1.0:

```txt
No
```

Relation to thesis:

Supports discussion of uncertainty-aware decision support.

Relation to application track:

May provide simple uncertainty/risk indicators for the demo.

---

## `06_thesis_figures.ipynb`

Purpose:

Generate thesis-ready tables and figures from verified experiment outputs.

Inputs:

```txt
research/results/tables/
research/results/figures/
research/results/logs/
```

Outputs:

```txt
research/results/figures/
research/results/tables/
```

Required dependencies:

```txt
pandas
numpy
matplotlib
duckdb
```

Mandatory for V1.0:

```txt
Yes
```

Relation to thesis:

Directly supports the final report figures, tables and empirical results.

Relation to application track:

May reuse demo-ready metrics or exported decision examples.

---

## V1.0 mandatory notebooks

The mandatory V1.0 notebooks are:

```txt
00_data_check.ipynb
01_finrl_baseline.ipynb
02_gymnasium_env.ipynb
03_baseline_comparison.ipynb
06_thesis_figures.ipynb
```

Optional stretch notebooks:

```txt
04_iqn_experiment.ipynb
05_uncertainty_proxy.ipynb
```

## Rule

Notebooks should produce small, reproducible outputs in `research/results/`.

Large datasets, model checkpoints, caches and temporary files must not be committed.
