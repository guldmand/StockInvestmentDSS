#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-guldmand/StockInvestmentDSS}"

echo "Using repo: $REPO"
echo "Updating selected NOW issue bodies only."
echo "No labels, milestones, project fields, status, roadmap, priority or percentage will be changed."
echo ""

echo "Checking GitHub auth..."
gh auth status

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat > "$TMP_DIR/issue_4.md" <<'EOF'
## Goal

Create the minimal DSS web app shell for the V1.0 PoC.

## Description

The frontend should provide a simple browser-based entry point for the StockInvestmentDSS application track.

For V1.0, the frontend should be kept deliberately simple: static HTML, CSS and vanilla JavaScript are enough. The purpose is to prove the local DSS flow, not to build a polished product UI.

The frontend should be compatible with a later Django/guldmand.com landing-page integration and should use Zero Sum Public only as visual/charting inspiration, not as a vendored dependency.

## Track

- Application track: yes, because this is the user-facing DSS shell.
- Research track: indirect, because the frontend may later display research outputs, model metadata, risk metrics and backtest evidence.
- Shared track: yes, because it must communicate with the backend API and reflect shared storage/runtime assumptions.

## Layer

- Fast layer: yes. The frontend is part of the near real-time decision support experience.
- Slow layer: no. The frontend must not run training, backtesting or notebooks.

## System Context

- External repos: Zero Sum Public is a reference for market dashboard/charting ideas only. Do not vendor or depend on it in V1.0.
- Data pipeline: frontend reads from backend API endpoints, not directly from DuckDB.
- Storage / guldNAS / DuckDB: frontend should display backend/runtime status but should not own storage access.
- Devices / infrastructure: must run on the local dev machine first and later through Docker Compose.
- Containers / Docker / k3s: should be easy to serve in a lightweight frontend container.
- Research / Application split: frontend belongs in `system/frontend/`, not `research/`.

## Implementation

Create a minimal frontend shell under:

```text
system/frontend/
```

Recommended V1.0 structure:

```text
system/frontend/
├── public/
│   ├── index.html
│   ├── css/
│   │   └── app.css
│   └── js/
│       └── app.js
```

The page should include:

- project title: StockInvestmentDSS
- short thesis/DSS description
- backend health status block
- DuckDB/runtime status block if available through backend API
- link/button to backend OpenAPI docs at `http://localhost:8000/docs`
- placeholder sections for:
  - investor profile
  - strategy
  - portfolio
  - decision output
  - risk output
  - audit/evidence

The frontend should call at least:

```text
GET http://localhost:8000/health
GET http://localhost:8000/config/runtime
```

No login implementation is required here unless it is trivial. Login/front-page verification belongs to issue #39.

## Test

- Start backend using Docker Compose.
- Open frontend locally.
- Verify the page renders.
- Verify the frontend can call backend `/health`.
- Verify the frontend can call backend `/config/runtime`.
- Verify the page links to `/docs`.
- Verify no heavy frontend framework is required for V1.0.
- No unit tests required for the first static shell.

## Acceptance Criteria

- `system/frontend/` contains a minimal web app shell.
- Front page renders in a browser.
- Frontend can call the backend health endpoint.
- Frontend can display non-secret runtime/config status.
- Frontend links to OpenAPI/Swagger docs.
- Structure is ready to be served by a frontend container.
- The shell does not depend on Next.js, React, FinRL or external cloned repos.
- Zero Sum Public is treated as inspiration/reference only.

## Notes

Keep this minimal. The goal is a working DSS shell that supports the PoC and future demo flow. Polish, advanced charting and full UI design come later.
EOF

cat > "$TMP_DIR/issue_28.md" <<'EOF'
## Goal

Document the V1.0 PoC container architecture.

## Description

The project now has a working backend container and a local Docker Compose setup. This task documents how the containerized application track is structured and how it relates to local development, guldNAS, DuckDB, future frontend serving and later k3s deployment.

The document should make the runtime architecture clear enough that a human developer or AI coding agent can continue implementation without mixing backend, frontend, research notebooks, training workers and storage responsibilities.

## Track

- Application track: yes, because containers are the runnable DSS application foundation.
- Research track: indirect, because research outputs may later be consumed by the application.
- Shared track: yes, because Docker/Compose, DuckDB paths, guldNAS and future k3s affect both tracks.

## Layer

- Fast layer: backend API, frontend shell, runtime config, health checks, decision support endpoints.
- Slow layer: future ingestion workers, feature workers, training jobs, backtests and model registry writers.

## System Context

- External repos: no external repositories are required to start the local app. External repo usage should remain documented in `external/`.
- Data pipeline: containers must use configured paths and not commit runtime data.
- Storage / guldNAS / DuckDB: local fallback is `system/runtime-data/`; canonical persistent storage is guldNAS.
- Devices / infrastructure: local dev first; Turing Pi/k3s later; GPU/cloud for heavy training.
- Containers / Docker / k3s: Docker Compose is the local runtime bridge toward k3s manifests.
- Research / Application split: backend/frontend containers belong to `system/`; notebooks and experiments belong to `research/`.

## Implementation

Create or update:

```text
docs/infrastructure/container-architecture.md
```

Recommended sections:

```text
1. Purpose
2. Current V1.0 container status
3. Local Docker Compose layout
4. Backend container
5. Frontend container target
6. Runtime data and DuckDB paths
7. guldNAS persistent storage context
8. Fast layer vs slow layer
9. Future workers
10. Turing Pi / k3s translation
11. What is intentionally not included in V1.0
```

Current known facts to include:

```text
Backend:
- location: system/backend/
- Dockerfile: system/backend/Dockerfile
- API framework: FastAPI
- docs: /docs and /openapi.json
- health: /health and /health/duckdb
- config: /config/runtime

Compose:
- location: system/docker-compose.yml
- backend reads system/.env
- backend mounts system/runtime-data to /app/runtime-data

Storage:
- local fallback: system/runtime-data/market_research.duckdb
- canonical NAS path: /mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

## Test

- Verify the document matches the current repo structure.
- Verify it references the backend Dockerfile and Compose file correctly.
- Verify it explains that slow-layer training is not part of backend startup.
- Verify it references guldNAS/DuckDB path decisions.
- No unit tests required.

## Acceptance Criteria

- `docs/infrastructure/container-architecture.md` exists.
- Local Docker Compose setup is described.
- Backend container role is described.
- Frontend container target is described.
- DuckDB/runtime-data volume strategy is described.
- guldNAS storage context is described.
- Future worker architecture is described briefly.
- GPU/cloud training node is described as separate from the Turing Pi cluster.
- The document clearly separates fast-layer runtime from slow-layer training.

## Notes

This is a documentation task, not a new implementation task. Keep it accurate, concise and aligned with the current PoC.
EOF

cat > "$TMP_DIR/issue_39.md" <<'EOF'
## Goal

Verify that the local PoC app starts and that the front page/login flow is usable.

## Description

This task is the local smoke test for the application track once the minimal frontend shell and backend API are available through Docker Compose.

The first version may use a simple password gate or demo login. The goal is not production authentication. The goal is to verify that a user can open the local DSS shell, pass the demo gate, and reach the dashboard/demo area.

## Track

- Application track: yes, because this validates the local app experience.
- Research track: indirect, because later dashboard content may show research/model outputs.
- Shared track: yes, because backend runtime config and DuckDB status should be visible or testable.

## Layer

- Fast layer: yes. This validates the online user-facing path.
- Slow layer: no. No model training, notebooks or backtesting should run during this test.

## System Context

- External repos: no external repo cloning required.
- Data pipeline: frontend should communicate with backend API, not directly with DuckDB.
- Storage / guldNAS / DuckDB: verify backend can access local runtime-data/DuckDB fallback.
- Devices / infrastructure: must work locally first through Docker Compose.
- Containers / Docker / k3s: test should use the local Compose setup where possible.
- Research / Application split: research notebooks are not required to verify the local app.

## Implementation

Use the current local runtime target:

```text
cd system
docker compose up --build
```

Expected local endpoints:

```text
Frontend: http://localhost:3000
Backend:  http://localhost:8000
Docs:     http://localhost:8000/docs
Health:   http://localhost:8000/health
Runtime:  http://localhost:8000/config/runtime
```

If the frontend is not yet in Compose, document the temporary manual frontend run command in the close comment.

The demo login/password gate can use values from:

```text
system/.env
DEMO_LOGIN_ENABLED=true
DEMO_USERNAME=demo
DEMO_PASSWORD=demo
```

## Test

Verify:

- `docker compose up --build` starts required services.
- Front page loads.
- Backend health endpoint responds.
- Frontend can show backend status.
- Login/password gate works if implemented.
- Failed login gives sensible feedback if implemented.
- Successful login opens dashboard/demo content if implemented.
- Runtime data path remains ignored by Git.
- No heavy RL training starts as part of local app startup.

## Acceptance Criteria

- Local app starts.
- Front page loads.
- Backend health status is reachable.
- Login/password gate works or is explicitly documented as deferred.
- Successful login opens the dashboard/demo area if implemented.
- Failed login gives sensible feedback if implemented.
- Local app can be stopped cleanly.
- Test result is documented in the issue close comment.

## Notes

This is a smoke-test task. Do not turn it into a production authentication task. The purpose is to prove that the local PoC can be opened and demonstrated.
EOF

cat > "$TMP_DIR/issue_78.md" <<'EOF'
## Goal

Define the slow-layer offline training architecture.

## Description

The slow layer is responsible for data ingestion, feature construction, RL training, uncertainty-aware experiments, backtesting, evaluation, model checkpoints and thesis evidence.

It must be separated from the fast user-facing backend. The DSS should provide near real-time decision support using available/cached data and model outputs. It should not retrain deep RL models during a live user request.

## Track

- Application track: indirect, because the application may consume model registry outputs, metrics and checkpoints.
- Research track: yes, because this is where FinRL, Gymnasium, IQN, uncertainty modeling, backtesting and evaluation belong.
- Shared track: yes, because slow-layer outputs feed DuckDB, reports and application evidence.

## Layer

- Fast layer: consumes selected slow-layer outputs only.
- Slow layer: yes. This task defines the offline/research/training pipeline.

## System Context

- External repos: FinRL, Gymnasium, ObjectRL, SDU_DataScienceTool and similar tools belong primarily here unless explicitly needed by the backend.
- Data pipeline: raw snapshots -> Parquet/CSV -> features -> DuckDB -> model registry -> reports/app outputs.
- Storage / guldNAS / DuckDB: heavy artifacts and persistent datasets should use guldNAS-backed paths; local fallback can be used for PoC.
- Devices / infrastructure: local dev for small tests; GPU box/cloud for heavy training; Turing Pi/k3s can orchestrate light jobs later.
- Containers / Docker / k3s: slow-layer workers may become separate containers later, but should not be required to start the local app.
- Research / Application split: slow-layer code and notebooks belong in `research/`, not inside the backend image.

## Implementation

Create or update one of:

```text
docs/architecture/slow-fast-layer-architecture.md
docs/architecture/slow-layer-training-architecture.md
```

The slow-layer document should cover:

```text
Inputs:
- raw API responses
- market prices
- FinRL-compatible datasets
- macro/fundamental/news inputs later

Processing:
- data validation
- point-in-time feature generation
- baseline experiments
- RL training
- distributional RL/IQN experiments
- uncertainty-aware evaluation
- walk-forward backtesting

Outputs:
- model checkpoints
- model registry metadata
- evaluation metrics
- backtest results
- thesis figures/tables
- selected decision-support artifacts for the fast layer
```

Recommended storage references:

```text
guldNAS:
- /mnt/nas/stockinvestmentdss/model-checkpoints/
- /mnt/nas/stockinvestmentdss/backtest-results/
- /mnt/nas/stockinvestmentdss/experiment-artifacts/

Local:
- research/results/
- system/runtime-data/
```

## Test

- Verify the document clearly states that live user requests must not trigger heavy RL training.
- Verify model checkpoint and registry paths are described.
- Verify training targets are separated from backend/frontend containers.
- Verify data flow supports point-in-time evaluation.
- No code tests required.

## Acceptance Criteria

- Training data flow is described.
- Feature generation role is described.
- Model checkpoint path is described.
- Model registry idea is described.
- GPU/cloud training target is described.
- Turing Pi/k3s role is described without overstating it as heavy GPU compute.
- The slow layer is explicitly not used for live user interaction.
- Relationship to fast-layer decision support is described.

## Notes

This is an architecture task. Keep it practical and aligned with the V1.0 PoC. Implementation of full training jobs comes later.
EOF

cat > "$TMP_DIR/issue_79.md" <<'EOF'
## Goal

Define the fast-layer online decision support architecture.

## Description

The fast layer is the user-facing DSS runtime path. It must provide immediate decision support using available data, cached features, model metadata, user constraints and portfolio state.

It should not perform heavy training, full backtesting or notebook execution during a user request.

## Track

- Application track: yes, because this defines the online DSS backend/frontend flow.
- Research track: indirect, because the fast layer may consume outputs from research experiments.
- Shared track: yes, because it depends on DuckDB, model registry metadata, risk outputs and audit/evidence paths.

## Layer

- Fast layer: yes. This task defines the online decision support path.
- Slow layer: consumed indirectly through precomputed artifacts and model outputs.

## System Context

- External repos: the fast backend should not require FinRL/Gymnasium/ObjectRL at startup unless explicitly needed later.
- Data pipeline: fast layer reads current/cached data, selected features and model registry outputs.
- Storage / guldNAS / DuckDB: use configurable DuckDB paths and local fallback first.
- Devices / infrastructure: local dev + Docker Compose first; k3s later.
- Containers / Docker / k3s: backend and frontend containers are the initial fast-layer runtime units.
- Research / Application split: fast layer belongs in `system/`, not `research/`.

## Implementation

Create or update one of:

```text
docs/architecture/slow-fast-layer-architecture.md
docs/architecture/fast-layer-decision-support.md
```

Describe the expected online request flow:

```text
User opens DSS frontend
-> selects/enters investor profile
-> selects strategy
-> enters or loads portfolio
-> backend reads current runtime config
-> backend reads latest available/cached features
-> backend applies user constraints and risk profile
-> backend returns decision alternatives
-> frontend shows recommendation, risk output and audit/evidence summary
```

Initial PoC decision outputs can be placeholders, but the architecture should prepare for:

```text
- buy / hold / sell recommendations
- risk score
- confidence / uncertainty output
- explanation/audit trail
- reference to data timestamp / point-in-time state
```

The document should mention current backend capabilities:

```text
- FastAPI API
- OpenAPI docs at /docs
- health endpoint
- DuckDB health check
- runtime config endpoint
```

## Test

- Verify the document states that no live retraining is required.
- Verify it explains how cached/latest features are used.
- Verify it places backend/frontend under `system/`.
- Verify it explains where audit/evidence should eventually be produced.
- No unit tests required.

## Acceptance Criteria

- No live retraining requirement is documented.
- Current portfolio state is used.
- Latest/cached features are used.
- User constraints and risk profile are applied.
- Decision alternatives are generated immediately.
- Backend/frontend roles are described.
- DuckDB/model registry/audit trail roles are described.
- Relationship to the slow layer is explained.

## Notes

This task defines how the DSS can be responsive while still being grounded in research outputs. It is central to the thesis argument.
EOF

cat > "$TMP_DIR/issue_163.md" <<'EOF'
## Goal

Create a lightweight frontend container and add it to Docker Compose.

## Description

The project now has a working backend container and local Compose setup. This task adds the frontend runtime unit so the local PoC can be opened as a browser application.

For V1.0, use a minimal static HTML/CSS/JavaScript frontend served from a lightweight container. Do not introduce Next.js, React, Svelte, Vue or Blazor unless a later task explicitly requires it.

The frontend should be compatible with a later Django/guldmand.com integration, where a public thesis landing page can link to the DSS demo.

## Track

- Application track: yes, because this is the user-facing app container.
- Research track: indirect, because the frontend may later show model/research outputs.
- Shared track: yes, because it depends on backend API, runtime config and Compose.

## Layer

- Fast layer: yes. The frontend is part of the near real-time DSS interface.
- Slow layer: no. The frontend container must not run training or notebooks.

## System Context

- External repos: Zero Sum Public is reference only for future charting/dashboard inspiration. Do not vendor it.
- Data pipeline: frontend communicates with backend API only.
- Storage / guldNAS / DuckDB: frontend does not access storage directly; it may display backend runtime/DuckDB status.
- Devices / infrastructure: local Docker Compose first; k3s later.
- Containers / Docker / k3s: this creates the second runtime container after the backend.
- Research / Application split: frontend belongs under `system/frontend/`.

## Implementation

Create:

```text
system/frontend/
├── Dockerfile
├── nginx.conf
└── public/
    ├── index.html
    ├── css/
    │   └── app.css
    └── js/
        └── app.js
```

Update:

```text
system/docker-compose.yml
```

Add a frontend service that:

```text
- builds from system/frontend/Dockerfile
- exposes FRONTEND_PORT, default 3000
- serves static files through nginx
- can call backend at http://localhost:8000 during local development
- depends on backend
```

The frontend should show at minimum:

```text
- StockInvestmentDSS title
- short PoC description
- backend health status
- runtime/DuckDB path status from /config/runtime
- link to backend Swagger/OpenAPI docs
- placeholder navigation for Strategy, Portfolio, Decisions and Risk
```

## Test

Run from `system/`:

```text
docker compose up --build
```

Verify:

```text
http://localhost:3000
http://localhost:8000/health
http://localhost:8000/docs
```

Confirm:

- frontend container starts
- backend container starts
- frontend page renders
- frontend can call backend `/health`
- frontend can call backend `/config/runtime`
- Docker Compose can be stopped with `docker compose down`
- no frontend node_modules or generated files are committed
- no heavy frontend framework is introduced

## Acceptance Criteria

- Frontend Dockerfile exists.
- Static frontend shell exists.
- Frontend service is added to Docker Compose.
- `docker compose up --build` starts backend and frontend.
- Frontend is available on `localhost:3000`.
- Backend remains available on `localhost:8000`.
- Frontend can display backend health/runtime status.
- Compose setup remains simple and does not require external repo cloning.
- Zero Sum Public remains reference-only.

## Notes

This completes the minimal local application runtime shape:

```text
frontend container -> backend API container -> runtime-data/DuckDB
```

Keep it simple. Advanced charts and Django/guldmand.com integration can come later.
EOF

for n in 4 28 39 78 79 163; do
  echo "Updating issue #$n ..."
  gh issue edit "$n" --repo "$REPO" --body-file "$TMP_DIR/issue_${n}.md"
done

echo ""
echo "Done."
echo "Updated bodies for issues: #4 #28 #39 #78 #79 #163"
echo "Skipped #43, #44 and #135 because their bodies are already detailed."
