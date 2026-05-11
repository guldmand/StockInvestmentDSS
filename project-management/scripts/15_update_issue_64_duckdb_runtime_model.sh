#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Update Issue #64
# Define DuckDB canonical database file path on guldNAS
# ============================================================
#
# Purpose:
#   Updates issue #64 body so it clearly states that:
#   - DuckDB is embedded / in-process
#   - guldNAS stores the .duckdb database file
#   - compute runs on local dev, backend containers, notebooks, workers,
#     Turing Pi/k3s, GPU box or cloud
#   - V1.0 should avoid multiple concurrent writers
#
# Safe behavior:
#   - Does not change labels
#   - Does not change project fields
#   - Does not close the issue
#   - Only updates the body of issue #64
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
ISSUE_NUMBER="64"

echo "Using repo: $GH_REPO"
echo "Updating issue #$ISSUE_NUMBER"
echo ""

echo "Checking GitHub auth..."
gh auth status
echo ""

TMP_FILE="$(mktemp)"
cleanup() {
  rm -f "$TMP_FILE"
}
trap cleanup EXIT

cat > "$TMP_FILE" <<'EOF'
## Goal

Define the canonical DuckDB database file path for the PoC and its local fallback path.

## Description

DuckDB is the central analytical database format for the PoC, but it is not used as a standalone database server in V1.0.

DuckDB is an embedded / in-process analytical database. The `.duckdb` database file is stored on persistent storage, while DuckDB itself runs inside the Python process, backend container, notebook, worker or training job that opens the file.

The project therefore needs:

- one canonical persistent DuckDB file path on guldNAS
- one local fallback DuckDB path for development
- environment variables that make the path configurable
- a clear runtime rule for how application and research code should access the file

This distinction is important because guldNAS is the storage layer, while local development machines, Turing Pi/k3s workers, backend containers, notebooks, GPU machines or cloud jobs are the compute layer.

## Track

- Application track: yes, because the backend and decision engine need a stable DuckDB connection path.
- Research track: yes, because notebooks and experiments must read/write reproducible datasets, features, metrics and results.
- Shared track: yes, because DuckDB is the common analytical bridge between application outputs and research evidence.

## Layer

- Fast layer: reads current/cached market data, features, portfolio state, strategy state, model registry entries, risk output and audit logs.
- Slow layer: writes ingested data, feature builds, backtest results, model metrics, experiment metadata and model registry outputs.

## System Context

- External repos: FinRL, Gymnasium, ObjectRL and other tools may produce outputs that are imported into DuckDB, but external code should not be stored in the DuckDB folder.
- Data pipeline: DuckDB should support point-in-time metadata, raw/curated references, features, portfolios, strategies, decisions, risk metrics, model registry metadata and audit logs.
- Storage / guldNAS / DuckDB: this task defines the canonical DuckDB file location on persistent NAS storage. guldNAS stores the database file; compute/query execution happens in the process that opens the file.
- Devices / infrastructure: local dev may use a local fallback path; k3s/Turing Pi workers and backend containers may use the NAS-mounted canonical file path when appropriate.
- Containers / Docker / k3s: DuckDB path must be controlled by environment variables so containers can switch between local runtime data and mounted NAS storage.
- Research / Application split: both tracks must refer to the same database concept, even when local copies, snapshots or exported files are used.

## DuckDB Runtime Model

DuckDB is not deployed as a long-running database service in V1.0.

The intended model is:

```text
Compute layer:
- local development machine
- backend container
- research notebook
- ingestion worker
- feature worker
- decision worker
- training job
- Turing Pi / k3s workload
- GPU box / cloud training job

Storage layer:
- guldNAS
- /mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

The process that opens the `.duckdb` file executes DuckDB queries in-process.

This means:

```text
DuckDB database file location = guldNAS
DuckDB execution/runtime      = local Python process, notebook, backend, worker or training job
```

## Implementation

Define environment variable names such as:

```env
DUCKDB_PATH=./runtime-data/market_research.duckdb
GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

Recommended canonical persistent DuckDB file path:

```text
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

Recommended local fallback path:

```text
system/runtime-data/market_research.duckdb
```

Document how the app and research code choose between paths:

```text
Local development:
  DUCKDB_PATH=./runtime-data/market_research.duckdb

Research notebooks from /research:
  DUCKDB_PATH=../system/runtime-data/market_research.duckdb

NAS / k3s / mounted storage:
  DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

For V1.0, the path decision can be simple:

```text
APP_ENV=local      → use local fallback path
APP_ENV=test/prod  → use mounted NAS path when available
```

## Test

- Verify that `/mnt/nas/stockinvestmentdss/duckdb/` exists on guldNAS.
- Verify that the local fallback folder `system/runtime-data/` is ignored by Git.
- Verify that `.duckdb` and `.duckdb.wal` files are ignored by Git.
- Verify that `system/.env.example` documents `DUCKDB_PATH`.
- Verify that `research/.env.example` documents a research-compatible `DUCKDB_PATH`.
- Verify that a minimal Python script can connect to the local fallback DuckDB path.
- If backend `db.py` is created in this task or a related task, add/update a minimal database connection test.

## Acceptance Criteria

- Canonical DuckDB database file path is documented.
- Local fallback DuckDB path is documented.
- `.env` variable names are defined.
- The distinction between DuckDB file storage and DuckDB runtime execution is documented.
- Backup/snapshot idea is noted.
- Path works conceptually for both application and research tracks.
- Path can be mounted into Docker Compose and later k3s workloads.
- V1.0 concurrency rule is documented.

## V1.0 Concurrency Rule

For V1.0, avoid multiple concurrent writers to the same DuckDB file.

Recommended rule:

```text
one writer at a time
multiple readers only where safe
```

If concurrent writes become necessary later, introduce one of:

- controlled writer service
- job queue
- scheduled ingestion window
- snapshot/export workflow
- separate read-only copies for notebooks or demo use

## Notes

DuckDB is not a replacement for raw file storage.

Raw API responses, Parquet and CSV exports should still exist for reproducibility and point-in-time traceability.

The expected data relationship is:

```text
raw-api-responses/
→ parquet/raw and/or csv/raw
→ parquet/curated and/or csv/curated
→ parquet/features and/or csv/features
→ duckdb/
→ research/results/
→ reports/
```

This task should not require creating a full production database schema. The purpose is to lock the path and runtime model before backend, ingestion and research notebooks start depending on DuckDB.
EOF

gh issue edit "$ISSUE_NUMBER" \
  --repo "$GH_REPO" \
  --body-file "$TMP_FILE"

echo ""
echo "Done."
echo "Issue #64 body has been updated."
echo ""
echo "No labels, project fields, status, roadmap, priority or percentage were changed."
