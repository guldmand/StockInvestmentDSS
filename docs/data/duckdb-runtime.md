# DuckDB Runtime Model

DuckDB is the central analytical database format for the PoC, but it is not used as a standalone database server in V1.0.

DuckDB is an embedded / in-process analytical database. The `.duckdb` database file is stored on persistent storage, while DuckDB itself runs inside the Python process, backend container, notebook, worker or training job that opens the file.

This document defines the canonical DuckDB file path, the local fallback path, the environment variable convention and the runtime rule for both application and research code.

---

## 1. Purpose

The purpose of this document is to lock the DuckDB runtime model before backend code, ingestion jobs, feature pipelines, decision workers, research notebooks and reinforcement learning experiments begin depending on a database path.

The key distinction is:

- guldNAS is the storage layer.
- Local development machines, Turing Pi / k3s workers, backend containers, notebooks, GPU machines and cloud jobs are the compute layer.

DuckDB database file location and DuckDB execution are therefore not the same thing.

---

## 2. Storage layer

The canonical persistent DuckDB file location is:

`/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

This path is intended for guldNAS or any environment where the NAS storage is mounted.

The folder should exist on guldNAS:

`/mnt/nas/stockinvestmentdss/duckdb/`

The database file itself is:

`market_research.duckdb`

---

## 3. Local fallback paths

For local development from the `/system` folder, use:

`./runtime-data/market_research.duckdb`

For research notebooks from the `/research` folder, use:

`../system/runtime-data/market_research.duckdb`

These local fallback paths are used when the NAS path is not mounted or when development should happen independently of the shared persistent NAS database file.

---

## 4. Environment variables

The application, notebooks and workers should primarily read the active DuckDB path from:

`DUCKDB_PATH`

Example for local development from `/system`:

`DUCKDB_PATH=./runtime-data/market_research.duckdb`

Example for research notebooks from `/research`:

`DUCKDB_PATH=../system/runtime-data/market_research.duckdb`

Example for NAS-mounted environments:

`DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

The canonical NAS reference path may also be documented as:

`GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

Application and research code should normally use `DUCKDB_PATH` as the runtime value. `GULDNAS_DUCKDB_PATH` is primarily a documented reference to the canonical persistent location.

---

## 5. Recommended `.env.example` values

### `system/.env.example`

For the application/system folder:

`APP_ENV=local`

`DUCKDB_PATH=./runtime-data/market_research.duckdb`

`GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

### `research/.env.example`

For notebooks and research code started from `/research`:

`DUCKDB_PATH=../system/runtime-data/market_research.duckdb`

`GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

---

## 6. Runtime model

DuckDB is not deployed as a long-running database service in V1.0.

DuckDB queries execute inside the process that opens the `.duckdb` file.

Examples of compute processes:

- local development script
- backend container
- research notebook
- ingestion worker
- feature worker
- decision worker
- training job
- Turing Pi / k3s workload
- GPU box / cloud job

This means that the DuckDB database file may live on guldNAS, while query execution happens inside the local Python process, backend container, worker, notebook or training job.

In short:

`DuckDB file storage = guldNAS`

`DuckDB runtime execution = process that opens the file`

---

## 7. V1.0 path rule

For V1.0, the path decision can be simple:

`APP_ENV=local` uses the local fallback path.

`APP_ENV=test` or `APP_ENV=prod` uses the mounted NAS path when available.

However, the preferred implementation is to let `.env` control the concrete path through `DUCKDB_PATH`.

This keeps backend code, notebooks and workers simple because they only need to read one active runtime value:

`DUCKDB_PATH`

---

## 8. Application and research split

Both the application track and research track must refer to the same database concept.

The application track needs a stable DuckDB path because the backend and decision engine need access to current and cached analytical data.

The research track needs a stable DuckDB path because notebooks and experiments must read and write reproducible datasets, features, metrics, backtest outputs and model evaluation results.

The shared track uses DuckDB as the common analytical bridge between application outputs and research evidence.

Local copies, snapshots and exported files may be used, but the canonical database concept should remain the same.

---

## 9. Fast layer and slow layer

The DuckDB database can support both fast and slow analytical data flows.

Fast layer examples:

- current market data
- cached market data
- features
- portfolio state
- strategy state
- model registry entries
- risk output
- decision output
- audit logs

Slow layer examples:

- ingested data
- feature builds
- backtest results
- model metrics
- experiment metadata
- model registry outputs
- research result tables

This does not mean all schema must be created in this task. This task only defines the path and runtime model.

---

## 10. V1.0 concurrency rule

For V1.0, avoid multiple concurrent writers to the same DuckDB file.

Recommended rule:

- one writer at a time
- multiple readers only where safe

If concurrent writes become necessary later, introduce one of the following patterns:

- controlled writer service
- job queue
- scheduled ingestion window
- snapshot/export workflow
- separate read-only copies for notebooks or demo use

This keeps the V1.0 PoC simple and reduces the risk of file locking, write conflicts or inconsistent analytical state.

---

## 11. Relationship to raw data

DuckDB is not a replacement for raw file storage.

Raw API responses, Parquet exports and CSV exports should still exist for reproducibility and point-in-time traceability.

Expected data relationship:

`raw-api-responses/`

`-> parquet/raw and/or csv/raw`

`-> parquet/curated and/or csv/curated`

`-> parquet/features and/or csv/features`

`-> duckdb/`

`-> research/results/`

`-> reports/`

DuckDB should support analytical querying across these stages, but the raw and curated files should remain available as reproducible source artifacts.

---

## 12. Point-in-time traceability

The DuckDB setup should support point-in-time metadata and reproducible decision support.

Relevant metadata may include:

- ingestion timestamp
- source timestamp
- market data timestamp
- data vendor/source
- transformation timestamp
- feature build version
- model version
- strategy version
- decision timestamp
- audit event id

The purpose is to make it possible to explain what was known at the time a decision, backtest or model evaluation was produced.

This helps reduce look-ahead bias and supports the thesis goal of transparent decision support.

---

## 13. Backup and snapshot idea

The canonical DuckDB file on guldNAS should be backed up or snapshotted.

A future backup/snapshot workflow may include:

- scheduled DuckDB file snapshots
- export to Parquet
- export of important tables to CSV for inspection
- timestamped research snapshots
- read-only demo copies
- model/result snapshot folders linked to report outputs

For V1.0, this document only notes the backup/snapshot requirement. The full implementation can be handled in a later infrastructure or data management task.

---

## 14. Docker and k3s compatibility

The DuckDB path must be configurable through environment variables so that Docker Compose and later k3s workloads can switch between local runtime data and mounted NAS storage.

Example container runtime value:

`DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb`

Example mounted storage path:

`/mnt/nas/stockinvestmentdss:/mnt/nas/stockinvestmentdss`

This task does not require a full Docker Compose or k3s implementation. It only defines the path convention that those environments should use later.

---

## 15. Git ignore requirements

Local runtime data and DuckDB database files should not be committed to Git.

The repository should ignore:

`system/runtime-data/`

`runtime-data/`

`*.duckdb`

`*.duckdb.wal`

`*.duckdb.tmp`

This prevents local database state, write-ahead logs and temporary files from being committed accidentally.

---

## 16. Minimal connection test

A minimal Python script may be used to verify that the configured DuckDB path works.

The test should:

- read `DUCKDB_PATH`
- create the parent folder if needed
- open a DuckDB connection
- create a small test table
- insert one row
- read the row back
- print the active database path

This verifies that the path is usable without introducing the full production database schema.

---

## 17. Acceptance criteria mapping

This document satisfies the following task decisions:

- Canonical DuckDB database file path is documented.
- Local fallback DuckDB path is documented.
- Environment variable names are defined.
- The distinction between DuckDB file storage and DuckDB runtime execution is documented.
- Backup/snapshot idea is noted.
- Path works conceptually for both application and research tracks.
- Path can be mounted into Docker Compose and later k3s workloads.
- V1.0 concurrency rule is documented.

This document does not create a full production database schema. Schema design should be handled in a later data model task.
