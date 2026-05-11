# DuckDB Parallel Compute and Single Writer Model

This note documents how the PoC can use four parallel k3s nodes without allowing multiple concurrent writers to the same DuckDB database file.

## Core idea

The PoC should not allow all nodes to write directly to the same `.duckdb` file.

Instead, use this model:

```txt
4 nodes = parallel compute
1 writer = controlled commit/import to DuckDB
```

The k3s nodes perform the heavy work in parallel. The DuckDB writer only performs the controlled registration/import step.

DuckDB is not used as a classic multi-writer database server in V1.0. The safe V1.0 rule is:

```txt
Parallel workers produce artifacts.
One writer registers/imports artifacts.
DuckDB queries artifacts and metadata.
```

Reference:

- DuckDB concurrency documentation: https://duckdb.org/docs/current/connect/concurrency.html

## Why k3s still makes sense

The k3s cluster is still useful because the expensive work can run in parallel across nodes.

Examples of parallel compute work:

```txt
fetch market data
call external APIs
build features
run backtests
train models
evaluate strategies
generate results
write Parquet/CSV/JSONL artifacts
```

The single DuckDB writer only handles the final controlled step:

```txt
read completed outputs
validate schema
append/import batches to DuckDB
register metadata
mark jobs as imported
```

This avoids write conflicts without removing the value of the cluster.

## V1.0 model

Do not design this:

```txt
node1 writes directly to DuckDB
node2 writes directly to DuckDB
node3 writes directly to DuckDB
node4 writes directly to DuckDB
```

Design this instead:

```txt
node1 -> writes Parquet/job output
node2 -> writes Parquet/job output
node3 -> writes Parquet/job output
node4 -> writes Parquet/job output

duckdb-writer -> imports/registers outputs in batches
```

The writer only becomes a bottleneck if the system produces extremely high write volume. For the thesis PoC, this is acceptable and much safer than multi-node direct writes to one DuckDB file.

## DuckDB does not need to import everything

DuckDB can query Parquet files directly.

Example:

```sql
SELECT *
FROM read_parquet('data/features/**/*.parquet');
```

This means workers can produce Parquet files in parallel, while DuckDB can act as the analytical query layer over those files.

Reference:

- DuckDB multiple files documentation: https://duckdb.org/docs/current/data/multiple_files/overview.html

## Recommended split

Use DuckDB for:

```txt
metadata
job registry
model registry
decision logs
summary tables
evaluation tables
small/medium curated analytical tables
```

Use files/Parquet for:

```txt
raw market data
large feature datasets
backtest outputs
training outputs
snapshots
```

## Practical PoC architecture

```txt
k3s workers
  -> produce artifacts on guldNAS
  -> write Parquet/CSV/JSONL/job outputs

duckdb-writer
  -> reads completed artifacts
  -> validates them
  -> imports or registers them in DuckDB

DuckDB
  -> stores metadata and selected analytical tables
  -> queries Parquet artifacts when useful
```

## Result

This gives the PoC:

```txt
parallel compute from k3s
reproducible artifacts on guldNAS
controlled DuckDB writes
DuckDB as analytical index/query layer
no multi-writer file conflicts
```

## V1.0 rule

For V1.0:

```txt
Only one process writes to market_research.duckdb.

Parallel workers write artifacts, not directly to DuckDB.

The DuckDB writer imports or registers completed artifacts in controlled batches.

Backend and notebooks can read from DuckDB, or from snapshots/read-only copies when needed.
```
