#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 14
# Update NOW issue bodies with operational architecture sections
# ============================================================
#
# Purpose:
#   Updates only the current NOW issues with consistent, actionable issue bodies.
#
# Sections used:
#   ## Goal
#   ## Description
#   ## Track
#   ## Layer
#   ## System Context
#   ## Implementation
#   ## Test
#   ## Acceptance Criteria
#   ## Notes
#
# Context:
#   - Application track + Research track
#   - Fast layer + Slow layer
#   - External repos
#   - Data pipeline
#   - guldNAS / DuckDB / Parquet / CSV
#   - Turing Pi / NAS / local dev / containers
#
# Safe behavior:
#   - Does not create issues
#   - Does not change labels
#   - Does not change project fields
#   - Only updates issue bodies for known NOW issue numbers
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"

echo "Using repo: $GH_REPO"
echo ""
echo "Checking GitHub auth..."
gh auth status
echo ""

update_issue_body() {
  local issue_number="$1"
  local tmp_file="$2"

  echo "Updating issue #${issue_number}..."
  gh issue edit "$issue_number" \
    --repo "$GH_REPO" \
    --body-file "$tmp_file"
}

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ------------------------------------------------------------
# #1 — Setup PoC repository structure
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_1.md" <<'EOF'
## Goal

Create the minimal but correct V1.0 PoC repository structure for StockInvestmentDSS.

## Description

This task establishes the foundation for the entire proof-of-concept. The repository must clearly separate the runnable decision support application from the academic research workspace and project documentation.

The structure must reflect the README v4.1 architecture:

```text
StockInvestmentDSS/
├─ docs/
├─ project-management/
├─ system/
├─ research/
├─ external/
├─ data/
└─ .github/
```

The key correction is that `system/` is not the repository root. It is the runnable DSS system inside the repository.

## Track

- Application track: yes, because `system/` will contain the runnable DSS app, backend, frontend, containers and local runtime setup.
- Research track: yes, because `research/` must exist for notebooks, experiments, results and report integration.
- Shared track: yes, because `docs/`, `external/`, `data/` and `.github/` support both tracks.

## Layer

- Fast layer: represented by the future app/backend/frontend structure inside `system/`.
- Slow layer: represented by `research/`, experiment outputs, FinRL/Gymnasium work and future training workers.

## System Context

- External repos: create or document `external/` as the place for external dependency manifests, not blindly vendored code.
- Data pipeline: reserve structure for raw data, DuckDB, Parquet and CSV outputs, but do not commit runtime data.
- Storage / guldNAS / DuckDB: local runtime data must be ignored; canonical long-term storage should be documented as guldNAS-backed.
- Devices / infrastructure: structure must support local dev first, then Turing Pi/k3s and NAS integration.
- Containers / Docker / k3s: structure must support Dockerfiles, docker-compose and later k3s manifests.
- Research / Application split: make it obvious where app code lives and where thesis experiments live.

## Implementation

Create or verify the top-level folders:

```text
docs/
project-management/
system/
research/
external/
data/
.github/
```

Create placeholder README files where useful:

```text
system/README.md
research/README.md
external/README.md
data/README.md
project-management/README.md
```

Ensure ignored runtime paths are represented in `.gitignore`, especially:

```text
.env
system/.env
system/runtime-data/
research/results/raw/
research/results/processed/
external/*/
*.duckdb
*.parquet
*.csv
```

Do not add large data files, model checkpoints or cloned external repositories unless explicitly required.

## Test

- Verify the repository can be opened cleanly from VS Code.
- Verify no unwanted runtime files are staged with `git status`.
- Verify expected folders exist.
- Add docstrings only when Python files are created as part of the task.
- No unit tests are required for pure folder scaffolding, but the structure should support later tests.

## Acceptance Criteria

- Top-level repository structure exists and matches README v4.1.
- `system/` exists for the runnable DSS application.
- `research/` exists for notebooks, experiments, report, figures and results.
- `external/` exists or is documented for external dependency manifests.
- `data/README.md` explains that large/runtime data is external and should not be committed.
- `.env.example` exists or is planned in the correct location.
- `system/runtime-data/` is ignored in Git.
- The project can proceed to Docker, DuckDB, FinRL and notebook setup without restructuring.

## Notes

This task is the structural foundation for the five-day PoC sprint. Keep it simple, clear and aligned with README v4.1.
EOF

# ------------------------------------------------------------
# #32 — Install and verify FinRL environment
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_32.md" <<'EOF'
## Goal

Install and verify that FinRL can run in the project environment without blocking the local PoC application.

## Description

FinRL is the primary financial RL framework for the thesis. This task verifies that the project can import and use FinRL at a minimal level before deeper training or backtesting tasks begin.

This is not a full training task. It is a foundation task for the slow-layer research pipeline and later application integration.

## Track

- Application track: indirect, because trained/evaluated outputs may later be consumed by the DSS backend and decision engine.
- Research track: yes, because FinRL belongs to experiments, notebooks, baseline comparisons and thesis evaluation.

## Layer

- Fast layer: no live retraining. The fast layer may later consume stored model outputs, risk metrics or model registry entries.
- Slow layer: yes. FinRL is part of offline training, backtesting, evaluation and research experiments.

## System Context

- External repos: FinRL must be treated as an external dependency and documented/pinned through the external dependency strategy.
- Data pipeline: FinRL data output should later be compatible with DuckDB, Parquet and CSV storage.
- Storage / guldNAS / DuckDB: experiment outputs should later be written to canonical storage or `research/results/`.
- Devices / infrastructure: test locally first; heavier training may later run on GPU box, cloud GPU, or scheduled infrastructure.
- Containers / Docker / k3s: installation notes should support later Docker worker images and not assume only one machine.
- Research / Application split: FinRL work belongs primarily in `research/` and `system/workers/finrl-worker/`, not inside the fast backend request path.

## Implementation

Create a minimal verification path, for example:

```text
research/notebooks/01_finrl_baseline.ipynb
or
research/experiments/finrl_baseline/smoke_test.py
```

Verify:

```python
import finrl
```

If direct `finrl` import is not stable, document the correct import paths and installed package versions.

Record:

- Python version
- package manager used
- installed FinRL version or commit
- known issues
- whether PyTorch is CPU-only or GPU-enabled
- whether the setup is local-only or container-ready

Do not start full RL training in this task.

## Test

- Run minimal import test.
- Run a small environment/data smoke test if feasible.
- Verify no FinRL cache, large dataset or model checkpoint is committed.
- Add small script or notebook cell that prints package versions.
- Document any failing dependency clearly instead of hiding it.

## Acceptance Criteria

- FinRL dependencies are installed in the active Python environment or documented as pending with a precise blocker.
- A minimal FinRL import test works or the exact failure is documented.
- Environment setup steps are documented.
- Known installation issues are written down.
- The setup does not block the local PoC app.
- External dependency handling is consistent with the `external/` strategy.
- The task produces enough information for a later Docker/worker setup.

## Notes

This task verifies the RL framework foundation. Full training, hyperparameter tuning and model registry integration belong to later slow-layer tasks.
EOF

# ------------------------------------------------------------
# #25 — Create Dockerfile for PoC backend
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_25.md" <<'EOF'
## Goal

Create a Dockerfile for the Python PoC backend/application service.

## Description

The backend container is the first reproducible runtime unit for the application track. It should run the FastAPI/Python backend locally and later support k3s deployment on the Turing Pi cluster.

This task should not mix slow-layer RL training into the backend image. The backend is primarily part of the fast layer and should stay lightweight.

## Track

- Application track: yes. This is part of the runnable DSS app.
- Research track: indirect. The backend may later expose research outputs, model registry entries or thesis demo data, but it should not run heavy notebooks or training directly.

## Layer

- Fast layer: yes. The backend supports near real-time decision support, health checks, stock lookup, portfolio/strategy flow and decision output.
- Slow layer: no, except for reading artifacts produced by slow-layer jobs, such as model registry metadata, metrics or checkpoints.

## System Context

- External repos: do not clone FinRL, Gymnasium, ObjectRL or SDU_DataScienceTool into the backend image unless explicitly needed.
- Data pipeline: backend should read configured DuckDB/runtime paths but not own the full ingestion/training pipeline.
- Storage / guldNAS / DuckDB: container must support configurable DuckDB path via environment variables.
- Devices / infrastructure: must run locally first and later be compatible with Turing Pi/k3s.
- Containers / Docker / k3s: this Dockerfile is the baseline for local Docker Compose and later deployment.
- Research / Application split: keep research notebooks and training dependencies out of this backend image unless required for a minimal demo.

## Implementation

Suggested location:

```text
system/backend/Dockerfile
```

or, if the current PoC is still a single-app skeleton:

```text
system/Dockerfile
```

The Dockerfile should:

- use an appropriate Python base image
- copy backend source code
- install from `requirements.txt`
- expose the backend port
- use environment variables for paths/config
- run the app with a clear command
- support fast rebuilds during local iteration

Example build command:

```bash
docker build -t stockinvestmentdss-backend ./system/backend
```

or if using a temporary single-service structure:

```bash
docker build -t stockinvestmentdss-poc ./system
```

## Test

- Build the image locally.
- Start the container.
- Confirm health endpoint responds, for example `/health`.
- Confirm `.env` variables can be passed through Docker Compose.
- Confirm the container can access the configured runtime-data path.
- Add or update Python unit tests where the Dockerfile depends on backend entrypoints.
- Add docstrings to new Python modules/functions created during this task.

## Acceptance Criteria

- Dockerfile exists for the Python backend/app.
- Image builds locally.
- App starts inside the container.
- Environment variables are loaded from `.env` or compose.
- Container can access the runtime-data path.
- DuckDB path is configurable.
- Image is small enough for fast local iteration.
- Dockerfile does not unnecessarily bundle heavy slow-layer training dependencies.

## Notes

This supports reproducibility, local smoke testing and future k3s deployment. Keep the backend image focused on serving the DSS application.
EOF

# ------------------------------------------------------------
# #26 — Create docker-compose.yml for local PoC
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_26.md" <<'EOF'
## Goal

Create a local Docker Compose setup for the V1.0 PoC.

## Description

Docker Compose should make the local application track easy to run from a clean checkout. It should start the backend/app and mount runtime data paths consistently.

This is the bridge between local development and later k3s deployment. It should be simple enough for the thesis demo and strict enough to support reproducibility.

## Track

- Application track: yes. Compose runs the local DSS application.
- Research track: indirect. It should expose shared runtime paths so research outputs can later be consumed by the app.
- Shared track: yes, because the same `.env` and storage assumptions affect both tracks.

## Layer

- Fast layer: yes. Compose should start the service that supports near real-time decision support.
- Slow layer: optional placeholder only. Training jobs or FinRL workers can be added later, but should not be required for the first app startup.

## System Context

- External repos: compose should not depend on cloned external repositories for the first app startup.
- Data pipeline: mount paths for DuckDB, raw files, Parquet and CSV as needed.
- Storage / guldNAS / DuckDB: support local fallback first; allow later guldNAS path through env variables.
- Devices / infrastructure: must work on development machine first.
- Containers / Docker / k3s: compose is the local equivalent of the later k3s service layout.
- Research / Application split: research notebooks should not be required to start the app.

## Implementation

Suggested location:

```text
system/docker-compose.yml
```

Minimum services:

```text
backend/app
```

Optional placeholders:

```text
ingestion-worker
feature-worker
decision-worker
finrl-worker
```

Only include optional workers if they do not complicate the first local startup.

The compose file should:

- build the backend/app image
- read `.env`
- mount local runtime-data
- expose backend port
- optionally expose frontend port if a frontend shell exists
- define named volumes or bind mounts clearly

Example command:

```bash
cd system
docker compose up --build
```

## Test

- Run `docker compose config`.
- Run `docker compose up --build`.
- Confirm backend health endpoint responds.
- Confirm runtime-data path is mounted.
- Confirm `.env` variables are visible inside the container.
- Add/update smoke test script if app startup is affected.
- No heavy RL training should run as part of compose startup.

## Acceptance Criteria

- `docker-compose.yml` exists.
- App/backend service starts successfully.
- Runtime-data volume/path is mounted.
- `.env` file is supported.
- DuckDB path is configurable.
- Command works: `docker compose up --build`.
- Compose setup does not require FinRL training or external repo cloning to start.
- Setup can later be translated to k3s manifests.

## Notes

This makes the PoC reproducible and supports the transition from local development to test deployment on the Turing Pi/k3s cluster.
EOF

# ------------------------------------------------------------
# #43 — Define report.tex structure
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_43.md" <<'EOF'
## Goal

Create the main LaTeX report structure for the thesis.

## Description

The report structure should match the thesis problem formulation and the V1.0 PoC architecture. It must support writing in parallel with implementation and research experiments.

The report should make the system design explicit: application track, research track, fast layer, slow layer, point-in-time data, auditability, FinRL/RL foundation and risk-aware evaluation.

## Track

- Application track: indirect, because the report must describe the DSS architecture, UI/demo flow and audit trail.
- Research track: yes, because the report is the final academic output.
- Shared track: yes, because it connects implementation, experiments, results and thesis argumentation.

## Layer

- Fast layer: report must describe near real-time decision support and why it does not retrain models per user action.
- Slow layer: report must describe offline training, backtesting, evaluation and model/metric generation.

## System Context

- External repos: report should have room to explain FinRL, Gymnasium, ObjectRL, SDU_DataScienceTool and any frontend reference source.
- Data pipeline: report must cover point-in-time ingestion, DuckDB, Parquet/CSV and raw snapshots.
- Storage / guldNAS / DuckDB: report should document reproducibility and storage assumptions.
- Devices / infrastructure: report should include local dev, guldNAS, Turing Pi/k3s and optional GPU/cloud roles at a high level.
- Containers / Docker / k3s: report should describe reproducible deployment and demo setup.
- Research / Application split: report should explain how notebooks/experiments and the app support the same thesis.

## Implementation

Suggested structure:

```text
research/report/
├─ report.tex
├─ references.bib
├─ sections/
│  ├─ 01_introduction.tex
│  ├─ 02_background.tex
│  ├─ 03_system_design.tex
│  ├─ 04_methodology.tex
│  ├─ 05_results.tex
│  ├─ 06_discussion.tex
│  └─ 07_conclusion.tex
├─ figures/
├─ tables/
└─ build.sh
```

The first version can contain empty sections with clear TODO markers.

Ensure the structure supports:

- problem formulation
- research questions
- related work
- system architecture
- methodology
- experiments
- results/case demonstration
- limitations
- conclusion/future work

## Test

- Run a minimal LaTeX build if tooling is available.
- Confirm `report.tex` can include all section files.
- Confirm figures/tables paths are valid.
- Confirm at least one citation can be compiled after `references.bib` exists.
- No Python unit tests required unless build scripts are added.

## Acceptance Criteria

- `research/report/report.tex` exists.
- Section files are defined.
- Figures and tables paths are defined.
- Compilation command is documented.
- Empty sections can be filled incrementally.
- Structure supports both application and research tracks.
- Structure has a place for slow/fast layer explanation.

## Notes

This should make thesis writing incremental instead of a final panic task. Keep the template simple and compatible with local LaTeX/Texifier workflow.
EOF

# ------------------------------------------------------------
# #44 — Define BibTeX reference file
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_44.md" <<'EOF'
## Goal

Create the BibTeX reference file for the thesis.

## Description

The thesis depends on a small set of core references covering reinforcement learning, distributional RL, uncertainty, FinRL, risk metrics and classical finance. This task creates a clean reference file early so the report can cite sources consistently from the beginning.

## Track

- Application track: indirect, because system design claims should be supported by literature when relevant.
- Research track: yes, because references support the academic framing, methodology and related work.

## Layer

- Fast layer: cite sources that justify decision support, risk-aware recommendations and no live retraining where relevant.
- Slow layer: cite sources for RL, FinRL, IQN, uncertainty modeling, backtesting and risk metrics.

## System Context

- External repos: references should include FinRL-related papers and later notes about external frameworks.
- Data pipeline: include point-in-time/backtesting and risk evaluation references where available.
- Storage / guldNAS / DuckDB: no direct citation requirement unless discussing reproducibility/data engineering.
- Devices / infrastructure: generally not citation-heavy unless discussing reproducible systems.
- Containers / Docker / k3s: no direct citation requirement for V1.0 unless included in system design.
- Research / Application split: references must support both the system design and the experimental methodology.

## Implementation

Create:

```text
research/report/references.bib
```

Add clean citation keys for at least:

- Dabney et al. — Implicit Quantile Networks
- Sensoy, Kaplan and Kandemir — Evidential Deep Learning
- FinRL paper(s)
- FinRL-Meta if used
- Markowitz portfolio selection
- Sharpe ratio / CAPM or Sharpe-related reference
- CVaR / Expected Shortfall reference if used
- Any major Gymnasium/ObjectRL references if used in text

Use stable citation keys, for example:

```text
dabney2018iqn
sensoy2018evidential
liu2020finrl
liu2022finrlmeta
markowitz1952portfolio
rockafellar2000cvar
```

## Test

- Confirm `report.tex` can cite at least one entry.
- Confirm BibTeX file parses without syntax errors.
- Confirm citation keys are consistent and readable.
- No unit tests required.

## Acceptance Criteria

- `research/report/references.bib` exists.
- Core RL, FinRL, IQN, uncertainty, CVaR and finance references are added.
- Citation keys are consistent.
- `report.tex` can cite at least one paper.
- The file supports the current problem formulation and V1.0 report structure.

## Notes

Do not overfill the bibliography with papers that are not used. Start with the core references needed for the problem formulation, background and methodology.
EOF

# ------------------------------------------------------------
# #62 — Create guldNAS storage folder structure
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_62.md" <<'EOF'
## Goal

Create the intended persistent storage folder structure on guldNAS.

## Description

Runtime data should not live permanently inside the Git repository. The PoC needs a canonical storage layout for DuckDB, raw API snapshots, Parquet, CSV, model checkpoints, logs, reports and experiment artifacts.

This task creates or documents the folder structure that both the application track and research track will share.

## Track

- Application track: yes, because the DSS backend and containers need stable data paths.
- Research track: yes, because notebooks and experiments need reproducible inputs/outputs.
- Shared track: yes, because guldNAS is the common storage layer.

## Layer

- Fast layer: uses DuckDB, cached features, latest known data, model metadata and audit logs.
- Slow layer: writes raw ingestions, curated datasets, features, model checkpoints, backtests and experiment outputs.

## System Context

- External repos: external code should not be stored here, but outputs from FinRL/Gymnasium experiments may be.
- Data pipeline: folders should support raw → curated → features → results flow.
- Storage / guldNAS / DuckDB: this is the canonical persistent storage task.
- Devices / infrastructure: guldNAS is the Raspberry Pi/NAS storage layer used by local dev and later k3s/Turing Pi.
- Containers / Docker / k3s: container mounts should eventually point to this structure.
- Research / Application split: both tracks must use shared paths without committing large files.

## Implementation

Create or document:

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
└── logs/
```

Also document local fallback paths:

```text
system/runtime-data/
research/results/
```

## Test

- Verify folder creation commands work on the target NAS mount.
- Verify local user has read/write permissions.
- Verify Docker/Compose can mount at least the local fallback path.
- Verify no generated data files are committed.
- Add a small README or `.gitkeep` only in local documentation folders if needed.

## Acceptance Criteria

- Root folder is defined.
- DuckDB folder is defined.
- Parquet raw/curated/features folders are defined.
- CSV raw/curated/features folders are defined.
- Raw API response folders are defined.
- Model-checkpoints folder is defined.
- Logs/results/report artifact folders are defined.
- Structure is documented in README or docs.
- Git ignores runtime data.

## Notes

The user explicitly wants data available in DuckDB, Parquet and CSV. DuckDB is the canonical analytical store; Parquet/CSV are export and reproducibility formats.
EOF

# ------------------------------------------------------------
# #64 — Define DuckDB canonical path on guldNAS
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_64.md" <<'EOF'
## Goal

Define the canonical DuckDB path for the PoC and its local fallback path.

## Description

DuckDB is the central analytical database for the PoC. The project needs one canonical path for persistent storage and one local fallback path for development.

The path must be configurable because the app may run locally, in Docker Compose, on the Turing Pi/k3s cluster, or later in another deployment environment.

## Track

- Application track: yes, because the backend and decision engine need a stable DuckDB connection.
- Research track: yes, because notebooks and experiments must read/write reproducible datasets and results.
- Shared track: yes, because DuckDB is the common bridge between app and research outputs.

## Layer

- Fast layer: reads current/cached market data, features, portfolio state, model registry entries and audit logs.
- Slow layer: writes ingested data, feature builds, backtest results, model metrics and experiment outputs.

## System Context

- External repos: FinRL and other tools may write outputs that are imported into DuckDB.
- Data pipeline: DuckDB should support point-in-time metadata, raw/curated references, features, portfolios, strategies, decisions and audit logs.
- Storage / guldNAS / DuckDB: this task defines the canonical database location.
- Devices / infrastructure: local dev should use fallback path; NAS/k3s should use mounted canonical path.
- Containers / Docker / k3s: path must be controlled by environment variable.
- Research / Application split: both tracks must refer to the same database concept, even if local copies are used.

## Implementation

Define environment variable names such as:

```text
DUCKDB_PATH=./runtime-data/market_research.duckdb
GULDNAS_DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

Recommended canonical path:

```text
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

Recommended local fallback:

```text
system/runtime-data/market_research.duckdb
```

Document how the app chooses between local and canonical paths.

## Test

- Verify a Python script can connect to the local fallback DuckDB path.
- Verify the folder exists before connection.
- Verify the database file is ignored by Git.
- Verify `.env.example` documents the path variable.
- Add/update a minimal database connection test if backend `db.py` is created.

## Acceptance Criteria

- Canonical path is documented.
- Local fallback path is documented.
- `.env` variable name is defined.
- Backup/snapshot idea is noted.
- Path works for both application and research tracks.
- Path can be mounted into Docker/Compose and later k3s.

## Notes

DuckDB is not a replacement for raw file storage. Raw API responses, Parquet and CSV exports should still exist for reproducibility and point-in-time traceability.
EOF

# ------------------------------------------------------------
# #115 — Define external repository dependency strategy
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_115.md" <<'EOF'
## Goal

Define how external repositories are referenced, pinned and documented in the PoC.

## Description

The project references several external repositories and frameworks:

- FinRL
- Gymnasium
- ObjectRL
- SDU_DataScienceTool
- Zero Sum Public

The PoC should not blindly vendor or clone large external repositories into the main source tree. Instead, the repository should contain a lightweight strategy that documents which external repositories are used, why they are used, and how they are pinned or fetched.

This supports reproducibility and makes it easier for Dockerfiles, setup scripts, notebooks and future AI coding agents to understand the dependency model.

## Track

- Application track: yes, because the DSS app may depend on selected external functionality or outputs.
- Research track: yes, because notebooks and experiments may rely on FinRL, Gymnasium, ObjectRL and SDU_DataScienceTool.
- Shared track: yes, because dependency documentation must serve both tracks.

## Layer

- Fast layer: should avoid heavy external RL dependencies unless only consuming outputs or lightweight adapters.
- Slow layer: may use FinRL, Gymnasium, ObjectRL and ML/RL tools for experiments, training and evaluation.

## System Context

- External repos: this is the main external dependency task.
- Data pipeline: SDU_DataScienceTool and FinRL may influence ingestion/data preparation.
- Storage / guldNAS / DuckDB: external dependency outputs should still flow into the canonical storage layout.
- Devices / infrastructure: external dependencies must be usable from local dev and later containers/workers.
- Containers / Docker / k3s: Docker builds may later fetch/pin these repos, but this task only defines the strategy.
- Research / Application split: the strategy should prevent research dependencies from unnecessarily bloating the fast application image.

## Implementation

Create or update:

```text
external/
├─ README.md
└─ external-repos.lock
```

Suggested `external-repos.lock` entries:

```text
FinRL=https://github.com/AI4Finance-Foundation/FinRL commit=<pin-later> role=financial-rl-framework
Gymnasium=https://github.com/Farama-Foundation/Gymnasium commit=<pin-later> role=environment-interface
ObjectRL=https://github.com/adinlab/objectrl commit=<pin-later> role=rl-prototyping-library
SDU_DataScienceTool=https://github.com/guldmand/SDU_DataScienceTool commit=<pin-later> role=existing-data-ingestion-tool
zero-sum-public=https://github.com/tristcoil/zero-sum-public commit=<pin-later> role=frontend-charting-reference
```

Do not clone external repositories as part of this task unless explicitly requested.

## Test

- Verify `external/README.md` explains the strategy clearly.
- Verify `external/external-repos.lock` is valid plain text or a simple parseable format.
- Verify `.gitignore` prevents accidental vendoring if needed.
- Verify README v4.1 remains consistent with this strategy.

## Acceptance Criteria

- `external/` folder is defined in the repository structure.
- `external/README.md` purpose is described.
- `external/external-repos.lock` or equivalent manifest is defined.
- FinRL, Gymnasium, ObjectRL, SDU_DataScienceTool and Zero Sum Public are listed with URL, role and pin field.
- README.md is consistent with this strategy.
- No external repository is cloned unless explicitly requested.

## Notes

This is a documentation and architecture task, not a heavy implementation task. The goal is clarity and reproducibility, not dependency-manager complexity.
EOF

# ------------------------------------------------------------
# #121 — Create research notebook index
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_121.md" <<'EOF'
## Goal

Create a clear index for all research notebooks used in the thesis PoC.

## Description

The project runs in two parallel tracks:

```text
Application track = working DSS application
Research track    = reproducible experiments, notebooks, figures and thesis outputs
```

The notebook index should explain the purpose of each notebook, expected inputs, expected outputs and how the results connect to the thesis report and application demo.

## Track

- Application track: indirect, because notebook outputs can become demo evidence, model metrics or decision-support artifacts.
- Research track: yes, this is the main notebook organization task.

## Layer

- Fast layer: notebook outputs may later feed cached features, baseline tables, model metadata or demo decision examples.
- Slow layer: notebooks document offline data checks, experiments, backtests, model runs, metrics and thesis figures.

## System Context

- External repos: notebooks may use FinRL, Gymnasium, ObjectRL and SDU_DataScienceTool, but each notebook should state its dependency level.
- Data pipeline: notebooks must respect raw → DuckDB → Parquet/CSV → features/results flow.
- Storage / guldNAS / DuckDB: notebooks should document whether they use local fallback or canonical NAS paths.
- Devices / infrastructure: notebooks should run on the dev machine first; heavy experiments may later be moved to GPU/cloud.
- Containers / Docker / k3s: notebooks are not part of the fast app container, but should support reproducible setup.
- Research / Application split: notebook outputs should be linkable to thesis tables/figures and optionally app demo data.

## Implementation

Create:

```text
research/notebooks/README.md
```

Document notebooks:

```text
00_data_check.ipynb
01_finrl_baseline.ipynb
02_gymnasium_env.ipynb
03_baseline_comparison.ipynb
04_iqn_experiment.ipynb
05_uncertainty_proxy.ipynb
06_thesis_figures.ipynb
```

For each notebook, include:

- purpose
- input paths
- output paths
- required dependencies
- whether mandatory for V1.0
- whether it supports fast layer, slow layer or both
- relation to thesis sections

## Test

- Verify notebook README exists.
- Verify listed notebooks match README v4.1.
- Verify output folders are documented.
- Verify mandatory V1.0 notebooks are clearly marked.
- No Python unit tests required, but notebook skeletons should be runnable later.

## Acceptance Criteria

- `research/notebooks/README.md` exists.
- Notebooks 00 to 06 are listed.
- Each notebook has purpose, inputs and outputs.
- Output folders for tables/figures are documented.
- README explains which notebooks are mandatory for V1.0.
- README explains how notebook outputs relate to the application track and thesis report.

## Notes

This index should make it easy for a human or coding agent to pick a notebook task and know exactly what it should produce.
EOF

# ------------------------------------------------------------
# #122 — Create notebook 00 data check skeleton
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_122.md" <<'EOF'
## Goal

Create the first data sanity-check notebook skeleton.

## Description

Notebook 00 should verify that the research environment is alive. It should check local paths, DuckDB connection, raw data folders, Parquet/CSV folders and basic market data availability.

This notebook should be the quickest way to confirm that the shared data foundation works before RL experiments, FinRL baselines or application integration begin.

## Track

- Application track: indirect, because it validates shared data assumptions used by the app/backend.
- Research track: yes, because it is the first reproducibility notebook.

## Layer

- Fast layer: validates paths and available data that fast decision support may later depend on.
- Slow layer: validates research/data availability before experiments, backtests and training.

## System Context

- External repos: should not require full FinRL training. It may optionally check whether FinRL is importable.
- Data pipeline: checks raw API folders, DuckDB, Parquet and CSV availability.
- Storage / guldNAS / DuckDB: must check canonical and local fallback paths when available.
- Devices / infrastructure: should run from development machine first.
- Containers / Docker / k3s: document whether the same checks can later run inside a container.
- Research / Application split: notebook should show which paths are shared with the application and which are research-only.

## Implementation

Create:

```text
research/notebooks/00_data_check.ipynb
```

Suggested notebook sections:

```text
1. Purpose
2. Environment variables
3. Local path checks
4. guldNAS path checks
5. DuckDB connection check
6. Raw API response folder check
7. Parquet folder check
8. CSV folder check
9. Minimal market data availability table
10. Summary / next actions
```

The notebook should run without requiring full RL training.

## Test

- Run the notebook top-to-bottom.
- Confirm it does not require unavailable API keys.
- Confirm it can connect to local fallback DuckDB or clearly explains missing DB.
- Confirm it checks both Parquet and CSV output paths.
- Confirm it does not write large files by accident.
- Add small helper functions with docstrings if Python utilities are created.

## Acceptance Criteria

- `research/notebooks/00_data_check.ipynb` exists.
- Notebook loads environment/config paths.
- Notebook checks DuckDB connection.
- Notebook checks raw/parquet/csv folder availability.
- Notebook includes a small data availability table.
- Notebook can run without requiring full RL training.
- Notebook distinguishes between local fallback and guldNAS canonical paths.

## Notes

This is a skeleton and sanity-check notebook. Keep it lightweight and reproducible.
EOF

# ------------------------------------------------------------
# #135 — Document two-track workflow: DSS app and research experiments
# ------------------------------------------------------------
cat > "$TMP_DIR/issue_135.md" <<'EOF'
## Goal

Document the two-track workflow that keeps the project coherent.

## Description

The thesis project must be developed as both:

```text
1. A working decision support system application
2. A reproducible research and thesis experiment environment
```

This task documents how `system/` and `research/` relate, what data they share, how slow-layer outputs can support fast-layer decision support, and how results move into the thesis report.

## Track

- Application track: yes, because `system/` must become a working DSS app.
- Research track: yes, because `research/` must produce reproducible experiments, metrics, figures and tables.
- Shared track: yes, because data, storage, external dependencies and audit evidence connect both tracks.

## Layer

- Fast layer: near real-time decision support using available data, cached features, model metadata and user constraints.
- Slow layer: offline ingestion, feature building, training, backtesting, evaluation and thesis reporting.

## System Context

- External repos: document which external dependencies belong mainly to research and which may affect the application.
- Data pipeline: define how data flows from raw/API sources into DuckDB, Parquet, CSV, features, experiments and app outputs.
- Storage / guldNAS / DuckDB: document shared canonical storage and local fallback paths.
- Devices / infrastructure: include MacBook/local dev, GPU box/cloud, Turing Pi/k3s and guldNAS roles.
- Containers / Docker / k3s: explain how application containers and future slow-layer workers relate.
- Research / Application split: make ownership of folders and outputs explicit.

## Implementation

Create or update one of:

```text
docs/architecture/two-track-workflow.md
docs/architecture/slow-fast-layer-architecture.md
README.md
research/README.md
system/README.md
```

The document should explain:

```text
Research track:
  notebooks
  experiments
  FinRL/Gymnasium
  model training
  backtesting
  thesis figures/tables

Application track:
  backend
  frontend
  strategy builder
  portfolio builder
  decision engine
  risk output
  audit log
  demo UI

Shared layer:
  DuckDB
  raw snapshots
  Parquet/CSV
  model registry
  external dependency manifest
  report evidence
```

## Test

- Verify the document is referenced from README or a relevant README.
- Verify terminology is consistent: fast layer, slow layer, application track, research track.
- Verify it explains what should not happen: no deep RL retraining in the fast user request path.
- No unit tests required unless supporting scripts are created.

## Acceptance Criteria

- Two-track workflow is documented.
- `system/` responsibility is described.
- `research/` responsibility is described.
- Shared data/storage assumptions are described.
- Flow from research result to thesis figure/table is described.
- Flow from system output to audit/evidence is described.
- Slow-layer / fast-layer separation is explained.
- External repos and containers are placed in the correct context.

## Notes

This document is important for AI-assisted development. It prevents future tasks from mixing notebooks, app code, heavy training and runtime decision support in the wrong places.
EOF

# ------------------------------------------------------------
# Apply updates
# ------------------------------------------------------------
update_issue_body 1 "$TMP_DIR/issue_1.md"
update_issue_body 32 "$TMP_DIR/issue_32.md"
update_issue_body 25 "$TMP_DIR/issue_25.md"
update_issue_body 26 "$TMP_DIR/issue_26.md"
update_issue_body 43 "$TMP_DIR/issue_43.md"
update_issue_body 44 "$TMP_DIR/issue_44.md"
update_issue_body 62 "$TMP_DIR/issue_62.md"
update_issue_body 64 "$TMP_DIR/issue_64.md"
update_issue_body 115 "$TMP_DIR/issue_115.md"
update_issue_body 121 "$TMP_DIR/issue_121.md"
update_issue_body 122 "$TMP_DIR/issue_122.md"
update_issue_body 135 "$TMP_DIR/issue_135.md"

echo ""
echo "Script 14 completed."
echo ""
echo "Updated NOW issue bodies:"
echo "- #1   Setup PoC repository structure"
echo "- #25  Create Dockerfile for PoC backend"
echo "- #26  Create docker-compose.yml for local PoC"
echo "- #32  Install and verify FinRL environment"
echo "- #43  Define report.tex structure"
echo "- #44  Define BibTeX reference file"
echo "- #62  Create guldNAS storage folder structure"
echo "- #64  Define DuckDB canonical path on guldNAS"
echo "- #115 Define external repository dependency strategy"
echo "- #121 Create research notebook index"
echo "- #122 Create notebook 00 data check skeleton"
echo "- #135 Document two-track workflow: DSS app and research experiments"
echo ""
echo "No labels, project fields, roadmap values or statuses were changed."
