#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 10
# Add missing README-alignment issues
#
# This script:
# - creates only missing README-alignment issues
# - does NOT modify existing issues
# - adds new issues to GitHub Project #11
# - sets project fields:
#   Status, Category, Priority, Roadmap, Track, Percentage,
#   Deadline, Progress Number
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"

PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

# ------------------------------------------------------------
# Read generated project.env safely if it exists
# ------------------------------------------------------------

read_env_value() {
  local key="$1"
  local file=".github/scripts-output/project.env"

  if [[ -f "$file" ]]; then
    grep "^${key}=" "$file" | head -n 1 | cut -d '=' -f2- | sed 's/^"//; s/"$//' || true
  fi
}

if [[ -f ".github/scripts-output/project.env" ]]; then
  OWNER_FROM_FILE="$(read_env_value "OWNER")"
  REPO_FROM_FILE="$(read_env_value "REPO")"
  GH_REPO_FROM_FILE="$(read_env_value "GH_REPO")"
  PROJECT_NUMBER_FROM_FILE="$(read_env_value "PROJECT_NUMBER")"
  PROJECT_TITLE_FROM_FILE="$(read_env_value "PROJECT_TITLE")"

  [[ -n "${OWNER_FROM_FILE:-}" ]] && OWNER="$OWNER_FROM_FILE"
  [[ -n "${REPO_FROM_FILE:-}" ]] && REPO="$REPO_FROM_FILE"
  [[ -n "${GH_REPO_FROM_FILE:-}" ]] && GH_REPO="$GH_REPO_FROM_FILE"
  [[ -n "${PROJECT_NUMBER_FROM_FILE:-}" ]] && PROJECT_NUMBER="$PROJECT_NUMBER_FROM_FILE"
  [[ -n "${PROJECT_TITLE_FROM_FILE:-}" ]] && PROJECT_TITLE="$PROJECT_TITLE_FROM_FILE"
fi

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE (#$PROJECT_NUMBER)"
echo ""

echo "Checking GitHub auth..."
gh auth status || {
  echo "Login first:"
  echo "  gh auth login --web"
  exit 1
}

# ------------------------------------------------------------
# Python detection
# ------------------------------------------------------------

PYTHON_BIN=""

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v py >/dev/null 2>&1; then
  PYTHON_BIN="py -3"
else
  echo "ERROR: Python not found."
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PYTHON_BIN" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

OWNER = "guldmand"
REPO = "StockInvestmentDSS"
GH_REPO = f"{OWNER}/{REPO}"
PROJECT_NUMBER = "11"
PROJECT_TITLE = "StockInvestmentDSS PoC Sprint"

env_path = Path(".github/scripts-output/project.env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip().strip('"')

        if key == "OWNER":
            OWNER = value
        elif key == "REPO":
            REPO = value
        elif key == "GH_REPO":
            GH_REPO = value
        elif key == "PROJECT_NUMBER":
            PROJECT_NUMBER = value
        elif key == "PROJECT_TITLE":
            PROJECT_TITLE = value

PERCENT_0 = "□□□□□□□□□□ 0%"

STATUS_TODO = "Todo"

CATEGORY_DEVELOPMENT = "⚙️ Development"
CATEGORY_DATA = "📊 Data"
CATEGORY_ARCHITECTURE = "🏗️ Architecture"

PRIORITY_HIGH = "⛰️ High"
PRIORITY_MEDIUM = "🫣 Medium"

ROADMAP_NEXT = "🔜 Next"
ROADMAP_LATER = "🗓️ Later"

TRACK_POC = "PoC"
TRACK_INFRA = "Infra"
TRACK_FUTURE = "Future Work"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run(cmd, *, check=True, capture=True):
    if capture:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        result = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if check and result.returncode != 0:
        print("")
        print("ERROR running command:")
        print(" ".join(cmd))
        if capture:
            print(result.stdout or "")
            print(result.stderr or "")
        sys.exit(result.returncode)

    return (result.stdout or "").strip() if capture else ""

def gh_json(cmd):
    out = run(cmd)
    if not out:
        return None
    return json.loads(out)

def check_graphql_rate_limit():
    data = gh_json([
        "gh", "api", "graphql",
        "-f", "query={ rateLimit { limit cost remaining used resetAt } }"
    ])

    rate = data["data"]["rateLimit"]

    print("")
    print("GraphQL rate limit:")
    print(f"  remaining: {rate['remaining']}")
    print(f"  used:      {rate['used']}")
    print(f"  resetAt:   {rate['resetAt']}")

    if rate["remaining"] < 300:
        print("")
        print("WARNING: GraphQL remaining is low.")
        print("Wait until resetAt if this script fails.")
        print("The script is idempotent and can safely be rerun.")

def create_label(name, color, description):
    result = subprocess.run([
        "gh", "label", "create", name,
        "--repo", GH_REPO,
        "--color", color,
        "--description", description,
        "--force",
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print(f"Label ready: {name}")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        print(f"Warning: could not create/update label {name}")
        print(msg)

def fetch_all_issues():
    data = gh_json([
        "gh", "api",
        "--paginate",
        "--slurp",
        f"repos/{GH_REPO}/issues?state=all&per_page=100"
    ]) or []

    issues = []

    for page in data:
        for item in page:
            if "pull_request" in item:
                continue

            issues.append({
                "number": item["number"],
                "title": item["title"],
                "url": item["html_url"],
                "labels": [label["name"] for label in item.get("labels", [])],
            })

    return issues

def issue_find_exact(title):
    for issue in fetch_all_issues():
        if issue["title"] == title:
            return issue
    return None

def create_or_get_issue(task):
    existing = issue_find_exact(task["title"])

    if existing:
        print(f"Issue already exists: {task['title']} (#{existing['number']})")
        return existing

    print(f"Creating issue: {task['title']}")

    cmd = [
        "gh", "issue", "create",
        "--repo", GH_REPO,
        "--title", task["title"],
        "--body", task["body"],
        "--label", ",".join(task["labels"]),
        "--milestone", task["milestone"],
    ]

    url = run(cmd)

    created = issue_find_exact(task["title"])
    if created:
        return created

    return {
        "number": None,
        "title": task["title"],
        "url": url,
        "labels": task["labels"],
    }

def add_issue_to_project(issue_url):
    if not issue_url:
        return

    result = subprocess.run([
        "gh", "project", "item-add", PROJECT_NUMBER,
        "--owner", OWNER,
        "--url", issue_url,
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print("  Added to project.")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        if "already exists" in msg.lower():
            print("  Already in project.")
        else:
            print("  Warning: could not add to project.")
            print(f"  {msg}")

def get_project_id():
    data = gh_json([
        "gh", "project", "view", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
    ])
    return data["id"]

def get_project_fields():
    data = gh_json([
        "gh", "project", "field-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "100",
        "--format", "json",
    ])
    return data["fields"]

def get_project_items():
    data = gh_json([
        "gh", "project", "item-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "300",
        "--format", "json",
    ])
    return data["items"]

def field_id(fields, name):
    for field in fields:
        if field.get("name") == name:
            return field.get("id")
    return None

def option_id(fields, field_name, option_name):
    for field in fields:
        if field.get("name") == field_name:
            for option in field.get("options", []):
                if option.get("name") == option_name:
                    return option.get("id")
    return None

def item_id_by_title(items, title):
    for item in items:
        content = item.get("content") or {}
        if content.get("title") == title:
            return item.get("id")
    return None

def update_single_select(project_id, item_id, field_id_value, option_id_value):
    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field_id_value,
        "--single-select-option-id", option_id_value,
    ])

def update_date(project_id, item_id, field_id_value, date_value):
    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field_id_value,
        "--date", date_value,
    ])

def update_number(project_id, item_id, field_id_value, number_value):
    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field_id_value,
        "--number", str(number_value),
    ])

def set_project_fields(project_id, item_id, fields, task):
    field_names = {
        "Status": task["status"],
        "Category": task["category"],
        "Priority": task["priority"],
        "Roadmap": task["roadmap"],
        "Track": task["track"],
        "Percentage": PERCENT_0,
    }

    for field_name, option_name in field_names.items():
        fid = field_id(fields, field_name)
        oid = option_id(fields, field_name, option_name)

        if not fid:
            print(f"  WARNING: Field not found: {field_name}")
            continue

        if not oid:
            print(f"  WARNING: Option not found: {field_name} -> {option_name}")
            continue

        update_single_select(project_id, item_id, fid, oid)
        print(f"  Set {field_name} = {option_name}")

    deadline_fid = field_id(fields, "Deadline")
    if deadline_fid:
        update_date(project_id, item_id, deadline_fid, task["deadline"])
        print(f"  Set Deadline = {task['deadline']}")
    else:
        print("  WARNING: Field not found: Deadline")

    progress_fid = field_id(fields, "Progress Number")
    if progress_fid:
        update_number(project_id, item_id, progress_fid, 0)
        print("  Set Progress Number = 0")
    else:
        print("  WARNING: Field not found: Progress Number")

# ------------------------------------------------------------
# Task definitions
# ------------------------------------------------------------

tasks = [
    {
        "title": "Create root .gitignore for runtime data, secrets and artifacts",
        "milestone": "M1 — PoC Foundation",
        "labels": ["documentation", "environment", "devops"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_INFRA,
        "status": STATUS_TODO,
        "deadline": "2026-05-11",
        "body": """## Goal

Create a root-level `.gitignore` that protects the repository from committing secrets, runtime data, large artifacts, generated files and local machine-specific files.

## Description

The README.md defines this repository as a thesis PoC with local runtime data, DuckDB files, Parquet/CSV exports, model checkpoints, logs, notebooks and generated report artifacts.

This task ensures that the repository can safely contain the project skeleton without accidentally committing:

- real `.env` files
- DuckDB runtime databases
- raw API responses
- Parquet/CSV exports
- model checkpoints
- logs
- LaTeX build artifacts
- Python caches
- notebook checkpoints
- local editor/OS files

## Acceptance Criteria

- root `.gitignore` exists
- `.env` files are ignored, but `.env.example` is allowed
- `system/runtime-data/` is ignored
- DuckDB database files are ignored
- Parquet/CSV/raw API exports are ignored
- model checkpoints and experiment artifacts are ignored
- Python cache folders are ignored
- notebook checkpoints are ignored
- LaTeX build artifacts are ignored
- OS/editor files are ignored
- repository can still keep `.gitkeep` and README files in otherwise ignored folders if needed

## AI Agent Instructions

Use README.md as the canonical architecture reference.

Do not invent a new repository structure.

Focus only on the root `.gitignore`.

Preserve the rule that runtime data should not live permanently inside Git.

Do not ignore documentation files, source code, SQL schema files, Docker files, GitHub Actions files or `.env.example`.

Prefer explicit ignore patterns with comments grouped by purpose.

After implementation, summarize which artifact categories are protected.

## Notes

This task supports reproducibility, security and clean thesis repository management.
"""
    },
    {
        "title": "Create root .env.example",
        "milestone": "M1 — PoC Foundation",
        "labels": ["documentation", "environment", "devops"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_INFRA,
        "status": STATUS_TODO,
        "deadline": "2026-05-11",
        "body": """## Goal

Create a root-level `.env.example` that documents the most important environment variables for local development, storage paths and PoC execution.

## Description

The system has both repository-level and system-level configuration needs.

The root `.env.example` should give a quick overview of the most important variables without exposing real secrets.

It should document:

- environment name
- backend/frontend ports
- app password/demo secret placeholder
- DuckDB path
- raw data path
- Parquet path
- CSV path
- model registry/checkpoint path
- log level
- optional guldNAS paths
- optional k3s/test/prod placeholders

## Acceptance Criteria

- root `.env.example` exists
- no real secrets are included
- local paths point to safe development defaults
- guldNAS paths are included as commented examples
- variables align with README.md storage layout
- variables can be copied into `.env` by a developer
- `.env` remains ignored by Git

## AI Agent Instructions

Use README.md as the canonical architecture reference.

Do not use real tokens, passwords, API keys or private paths.

Use placeholder values only.

Keep this file readable and grouped by section.

Do not remove the need for `system/.env.example`; this root file is an overview/template for repository-level setup.

After implementation, include a short note explaining how to copy it to `.env`.

## Notes

This task supports local reproducibility and prevents secrets from being committed.
"""
    },
    {
        "title": "Create system README",
        "milestone": "M1 — PoC Foundation",
        "labels": ["documentation", "backend", "frontend", "docker"],
        "category": CATEGORY_DEVELOPMENT,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "status": STATUS_TODO,
        "deadline": "2026-05-11",
        "body": """## Goal

Create `system/README.md` as the operational runbook for the runnable DSS PoC.

## Description

The root README explains the full thesis repository and architecture.

`system/README.md` should explain only the runnable system:

- how to start the local PoC
- expected services
- folder purpose
- environment variables
- Docker Compose usage
- backend health check
- runtime-data behavior
- where DuckDB is stored locally
- how this later maps to k3s/guldNAS

## Acceptance Criteria

- `system/README.md` exists
- local run command is documented
- Docker Compose command is documented
- backend and frontend URLs are documented
- health check command is documented
- runtime-data folder behavior is explained
- relation to guldNAS canonical storage is explained
- V1.0 scope is clear
- not-yet-implemented parts are marked honestly as planned/future

## AI Agent Instructions

Use root README.md as the canonical architecture reference.

Do not duplicate the entire root README.

Keep this file practical and operational.

Prefer commands and short explanations.

Do not claim features are implemented unless they exist.

If the current repo only has a skeleton, document the intended command and mark incomplete parts clearly.

## Notes

This is the file a developer or AI coding agent should read before working inside `system/`.
"""
    },
    {
        "title": "Create data README explaining external storage policy",
        "milestone": "M1 — PoC Foundation",
        "labels": ["documentation", "data", "guldnas"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "status": STATUS_TODO,
        "deadline": "2026-05-11",
        "body": """## Goal

Create `data/README.md` explaining that large/runtime data is stored outside Git and that guldNAS is the canonical persistent storage target.

## Description

The repository may contain a `data/` folder for documentation, but not for large datasets.

The README should explain the intended multi-format storage strategy:

- raw API responses
- DuckDB canonical analytical store
- Parquet datasets for ML/analytics
- CSV exports for debugging, inspection and thesis tables
- model checkpoints outside Git
- experiment artifacts outside Git

## Acceptance Criteria

- `data/README.md` exists
- states that large/runtime data must not be committed
- explains guldNAS as canonical persistent storage
- explains local development fallback under `system/runtime-data/`
- explains DuckDB, Parquet and CSV roles
- explains raw API responses as source evidence
- includes example folder layout
- references `.gitignore` and `.env.example`

## AI Agent Instructions

Use README.md as the canonical architecture reference.

Do not create large data files.

Do not add sample market datasets unless explicitly asked.

Do not commit generated DuckDB, Parquet, CSV, raw API, model checkpoint or log files.

Keep this README focused on policy and storage layout.

## Notes

CSV is allowed as a human-readable export/debug/reporting format, but it should not be treated as the primary source of truth.
"""
    },
    {
        "title": "Define optional GraphQL layer for PoC",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["documentation", "backend", "frontend", "architecture"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_FUTURE,
        "status": STATUS_TODO,
        "deadline": "2026-05-17",
        "body": """## Goal

Define whether and how the optional GraphQL layer should be used in the PoC.

## Description

README.md includes a possible GraphQL layer in both frontend and backend:

- frontend `GraphQLClientService.cs`
- backend `graphql/schema.py`
- backend `graphql/queries.py`
- backend `graphql/mutations.py`

However, GraphQL is not required for V1.0 if REST endpoints are sufficient.

This task should prevent architectural confusion by documenting the decision:

- use REST only for V1.0
- keep GraphQL as placeholder/future work
- or implement a minimal GraphQL schema later

## Acceptance Criteria

- GraphQL decision is documented
- V1.0 requirement is clarified
- if GraphQL is deferred, it is clearly marked as future work
- if placeholder files are created, they do not block REST implementation
- README/system docs are updated if needed
- no unnecessary GraphQL complexity is introduced into V1.0

## AI Agent Instructions

Use README.md as the canonical architecture reference.

Do not implement a full GraphQL layer unless explicitly requested.

Prefer REST endpoints for V1.0 if this keeps the PoC simpler.

If creating placeholder files, make them minimal and clearly documented.

Do not let GraphQL block data ingestion, strategy builder, portfolio builder or decision engine work.

## Notes

This task exists because README.md documents GraphQL, but the V1.0 PoC should stay focused.
"""
    },
    {
        "title": "Define backend route service repository layering convention",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["documentation", "backend", "architecture"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "status": STATUS_TODO,
        "deadline": "2026-05-11",
        "body": """## Goal

Define the backend layering convention used by the FastAPI PoC.

## Description

README.md describes a backend structure with:

- `api/` routes
- `services/` business logic
- `repositories/` DuckDB/database access
- `models/` DTOs
- `schemas/` validation schemas

This task documents the practical convention so future coding tasks and AI agents do not invent inconsistent patterns.

## Acceptance Criteria

- backend layering convention is documented
- responsibility of routes is defined
- responsibility of services is defined
- responsibility of repositories is defined
- responsibility of DTOs/schemas is defined
- naming convention is documented
- one example flow is described, such as:
  - route receives request
  - service validates business logic
  - repository reads/writes DuckDB
  - DTO/schema returns response
- documentation is placed in `docs/architecture/` or `system/backend/README.md`

## AI Agent Instructions

Use README.md as the canonical architecture reference.

Do not implement all backend modules in this task unless explicitly asked.

This task is primarily documentation and convention-setting.

Keep the convention practical for a fast V1.0 PoC.

Avoid enterprise overengineering.

Prefer clear files and simple imports over complex abstractions.

## Notes

This task will make later backend coding issues easier for GitHub Copilot or other AI coding agents to execute consistently.
"""
    },
]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

print("")
print("Checking GraphQL rate before running...")
check_graphql_rate_limit()

print("")
print("Creating required labels...")
required_labels = [
    ("documentation", "0075CA", "README, docs, runbooks"),
    ("environment", "C2E0C6", ".env files and environment configuration"),
    ("devops", "5319E7", "CI/CD, deployment workflows and automation"),
    ("backend", "D4C5F9", "Backend, API, app logic"),
    ("frontend", "BFD4F2", "Frontend, UI, charts, dashboard"),
    ("docker", "2496ED", "Dockerfiles, Docker Compose and container setup"),
    ("data", "C5DEF5", "Data pipeline, DuckDB, yfinance, features"),
    ("guldnas", "5319E7", "guldNAS Raspberry Pi persistent storage"),
    ("architecture", "5319E7", "Architecture and system design"),
]

for name, color, description in required_labels:
    create_label(name, color, description)

print("")
print("Fetching project metadata...")
project_id = get_project_id()
fields = get_project_fields()

print("")
print("Creating/finding missing README-alignment issues...")

created_or_found = []

for task in tasks:
    issue = create_or_get_issue(task)
    add_issue_to_project(issue.get("url"))
    created_or_found.append(task["title"])
    time.sleep(0.15)

time.sleep(2)

print("")
print("Fetching project items...")
items = get_project_items()

print("")
print("Setting project fields for created/found issues...")

for task in tasks:
    item_id = item_id_by_title(items, task["title"])

    if not item_id:
        print(f"WARNING: Project item not found for: {task['title']}")
        continue

    print("")
    print(f"Updating fields: {task['title']}")

    set_project_fields(
        project_id=project_id,
        item_id=item_id,
        fields=fields,
        task=task,
    )

    time.sleep(0.15)

print("")
print("Final GraphQL rate check...")
check_graphql_rate_limit()

print("")
print("Done.")
print("")
print("Script 10 completed.")
print("")
print("Created/found README-alignment issues:")
for title in created_or_found:
    print(f"- {title}")
print("")
print("No existing issue bodies were modified.")
PY