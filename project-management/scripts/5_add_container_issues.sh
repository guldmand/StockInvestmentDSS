#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 5
# Add missing Docker / container / k3s / worker issues
#
# Adds issues to:
#   guldmand/StockInvestmentDSS
#
# Adds them to project:
#   StockInvestmentDSS PoC Sprint
#
# Does NOT modify existing issues.
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE"

echo "Checking GitHub auth..."
gh auth status || {
  echo "Login first:"
  echo "  gh auth login --web"
  exit 1
}

create_label_if_missing() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label list --repo "$GH_REPO" --limit 200 --json name --jq '.[].name' | grep -Fxq "$name"; then
    echo "Label already exists: $name"
  else
    gh label create "$name" \
      --repo "$GH_REPO" \
      --color "$color" \
      --description "$description" >/dev/null

    echo "Created label: $name"
  fi
}

issue_exists() {
  local title="$1"

  gh issue list \
    --repo "$GH_REPO" \
    --state all \
    --search "$title in:title" \
    --json title \
    --jq '.[].title' | grep -Fxq "$title"
}

create_issue() {
  local title="$1"
  local milestone="$2"
  local labels="$3"
  local body="$4"

  if issue_exists "$title"; then
    echo "Issue already exists: $title"
    return 0
  fi

  echo "Creating issue: $title"

  gh issue create \
    --repo "$GH_REPO" \
    --title "$title" \
    --body "$body" \
    --label "$labels" \
    --milestone "$milestone" \
    --project "$PROJECT_TITLE" >/dev/null

  echo "  Created and added to project."
}

echo ""
echo "Ensuring clean labels exist..."

create_label_if_missing "docker" "2496ED" "Dockerfiles, Docker Compose and container setup"
create_label_if_missing "k3s" "5319E7" "k3s and Turing Pi deployment target"
create_label_if_missing "worker" "D4C5F9" "Background workers and scheduled jobs"
create_label_if_missing "training" "FBCA04" "Training jobs and model execution"
create_label_if_missing "infra" "5319E7" "Infrastructure, k3s, Turing Pi, deployment"
create_label_if_missing "backend" "D4C5F9" "Backend, API, app logic"
create_label_if_missing "data" "C5DEF5" "Data pipeline, DuckDB, yfinance, features"
create_label_if_missing "documentation" "0075CA" "README, docs, runbooks"
create_label_if_missing "poc" "0E8A16" "Required for PoC"
create_label_if_missing "bonus" "7057FF" "Bonus if time allows"
create_label_if_missing "future-work" "C5DEF5" "Future work / perspectives"
create_label_if_missing "urgent" "B60205" "Must be done immediately"
create_label_if_missing "high" "D93F0B" "Important for PoC"
create_label_if_missing "medium" "FBCA04" "Useful but not blocking"

echo ""
echo "Creating missing container/platform issues..."

create_issue \
"Create Dockerfile for PoC backend" \
"M1 — PoC Foundation" \
"docker,infra,backend,poc,urgent" \
"## Goal
Create a Dockerfile for the Python PoC backend/app.

## Acceptance criteria
- Dockerfile exists for the Python backend/app
- image builds locally
- app starts inside the container
- environment variables are loaded from .env
- container can access the runtime-data path
- image is small enough for fast local iteration

## Suggested location
- system/Dockerfile
- or app/Dockerfile depending on final PoC structure

## Example command
\`\`\`bash
docker build -t stockinvestmentdss-poc ./system
\`\`\`

## Thesis relevance
This supports reproducibility and makes the PoC easier to run, demonstrate and later deploy to k3s."

create_issue \
"Create docker-compose.yml for local PoC" \
"M1 — PoC Foundation" \
"docker,infra,poc,urgent" \
"## Goal
Create a local Docker Compose setup for the PoC.

## Acceptance criteria
- docker-compose.yml exists
- app service starts successfully
- runtime-data volume is mounted
- .env file is supported
- DuckDB path is configurable
- command works: docker compose up --build

## Minimum services
- app
- optional worker placeholder
- optional volume mount for runtime-data

## Example command
\`\`\`bash
docker compose up --build
\`\`\`

## Thesis relevance
This makes the PoC reproducible and supports the transition from local development to k3s deployment."

create_issue \
"Create Docker volume/storage plan for DuckDB" \
"M1 — PoC Foundation" \
"docker,data,infra,poc,high" \
"## Goal
Define how DuckDB is stored locally, in containers and later on NAS/k3s.

## Acceptance criteria
- DuckDB file path is defined
- local runtime-data path is ignored by Git
- container can read/write DuckDB
- NAS/k3s storage target is documented
- backup/snapshot idea is briefly described

## Suggested document
docs/infrastructure/duckdb-storage-plan.md

## Important note
DuckDB is embedded and file-based. For the PoC, it should be treated as a local analytical data store, not as a multi-user database server.

## Thesis relevance
This supports point-in-time storage, reproducibility and auditability."

create_issue \
"Document container architecture" \
"M1 — PoC Foundation" \
"documentation,infra,docker,poc,high" \
"## Goal
Document the PoC container architecture.

## Acceptance criteria
- docs/infrastructure/container-architecture.md exists
- local Docker Compose setup is described
- k3s deployment target is described
- DuckDB storage strategy is described
- worker architecture is described briefly
- GPU training node is described as separate from the Turing Pi cluster

## Suggested sections
1. Local development
2. Docker Compose PoC
3. DuckDB/runtime-data volume
4. Turing Pi/k3s target
5. GPU training node
6. Future worker architecture

## Thesis relevance
This connects the system architecture to the practical implementation of the decision support system."

create_issue \
"Create k3s deployment manifests for PoC app" \
"M6 — Buffer / v1.1" \
"k3s,infra,docker,bonus,medium" \
"## Goal
Create minimal k3s manifests for deploying the PoC app on the Turing Pi cluster.

## Acceptance criteria
- namespace manifest exists
- deployment manifest exists
- service manifest exists
- config/env handling is documented
- deploy command is documented
- storage limitation is documented

## Suggested location
infra/k3s/

## Example files
- namespace.yaml
- app-deployment.yaml
- app-service.yaml
- configmap.yaml

## Stop rule
This is bonus/v1.1. Do not work on this before the local PoC works.

## Thesis relevance
This demonstrates the intended deployment architecture without making Kubernetes the core thesis deliverable."

create_issue \
"Create worker container placeholder" \
"M6 — Buffer / v1.1" \
"worker,docker,infra,bonus,medium" \
"## Goal
Create a placeholder for background worker jobs.

## Acceptance criteria
- worker folder or module exists
- worker can be started separately
- worker has access to the same runtime-data volume
- worker role is documented
- worker does not block PoC v1.0

## Example future jobs
- scheduled yfinance ingestion
- feature generation
- market snapshot creation
- audit-log processing
- model output refresh

## Suggested location
system/worker/

## Thesis relevance
This supports the full-system architecture where ingestion and processing are separated from the user-facing decision support interface."

create_issue \
"Create training job container placeholder" \
"M6 — Buffer / v1.1" \
"training,docker,rl,infra,bonus,medium" \
"## Goal
Create a placeholder structure for offline RL training jobs.

## Acceptance criteria
- training job folder exists
- Dockerfile or run script is planned
- GPU/CUDA execution is documented as external to Turing Pi
- model checkpoint output path is described
- model registry idea is documented briefly

## Suggested location
system/training/

## Important architecture note
Turing Pi/k3s is used for orchestration and services. CUDA/RL training should run on the GPU machine or cloud GPU target.

## Thesis relevance
This documents the split between offline model training and online decision support."

echo ""
echo "Done."
echo ""
echo "Missing Docker/container/platform issues have been created and added to:"
echo "  $PROJECT_TITLE"
echo ""
echo "Recommended manual project field values:"
echo ""
echo "Now:"
echo "  Create Dockerfile for PoC backend"
echo "  Create docker-compose.yml for local PoC"
echo "  Create Docker volume/storage plan for DuckDB"
echo "  Document container architecture"
echo ""
echo "Later:"
echo "  Create k3s deployment manifests for PoC app"
echo "  Create worker container placeholder"
echo "  Create training job container placeholder"