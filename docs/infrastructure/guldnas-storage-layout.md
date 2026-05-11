# guldNAS Storage Layout

Canonical persistent storage for StockInvestmentDSS is located on guldNAS.

## Canonical Root

```text
/mnt/nas/stockinvestmentdss
```

## Canonical DuckDB File Path

```text
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

## Local Fallback Path

For local development, the DSS may use:

```text
system/runtime-data/market_research.duckdb
```

For research notebooks, the local fallback may be referenced as:

```text
../system/runtime-data/market_research.duckdb
```

## DuckDB Runtime Model

DuckDB is not deployed as a standalone database server in this PoC.

The DuckDB database file is stored on guldNAS for persistence, but DuckDB itself runs in-process inside the Python process that opens the file.

This means that DuckDB queries may be executed by:

```text
- local development scripts
- backend containers
- research notebooks
- ingestion workers
- feature workers
- decision workers
- training jobs
- Turing Pi / k3s workloads
- GPU box / cloud jobs
```

guldNAS is the storage layer.

The local dev machine, Turing Pi/k3s cluster, GPU box or cloud jobs are the compute layer.

## Environment Variables

Recommended local development value:

```env
DUCKDB_PATH=./runtime-data/market_research.duckdb
```

Recommended canonical NAS value:

```env
GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

When running from `/research`, the local fallback can be:

```env
DUCKDB_PATH=../system/runtime-data/market_research.duckdb
```

## Concurrency Rule for V1.0

For V1.0, avoid multiple concurrent writers to the same DuckDB file.

Recommended rule:

```text
one writer at a time
multiple readers only where safe
```

If concurrent writes become necessary later, introduce a controlled writer service, queue, scheduled ingestion window or snapshot/export workflow.

## Folder Layout

```text
/mnt/nas/stockinvestmentdss/
├── duckdb/
│   └── market_research.duckdb
├── parquet/
│   ├── raw/
│   ├── curated/
│   └── features/
├── csv/
│   ├── raw/
│   ├── curated/
│   └── features/
├── raw-api-responses/
│   ├── yfinance/
│   ├── finrl/
│   ├── gdelt/
│   ├── macro/
│   └── fundamentals/
├── model-checkpoints/
│   ├── finrl/
│   ├── iqn/
│   └── baselines/
├── backtest-results/
├── experiment-artifacts/
├── reports/
├── logs/
└── tmp/
```

## Verification

Verified on 2026-05-11.

```text
/mnt/nas/stockinvestmentdss exists
write access OK
RAID1 storage mounted at /mnt/nas
Samba share [nas] points to /mnt/nas
```

## Git Policy

Do not commit generated data files or runtime database files.

The repository should ignore:

```text
system/runtime-data/
*.duckdb
*.duckdb.wal
*.parquet
*.csv
```
