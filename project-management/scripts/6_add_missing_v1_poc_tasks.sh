#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 6
# Add missing V1.0 PoC support tasks
#
# Creates missing issues for:
# - FinRL
# - Gymnasium environment
# - guldNAS persistent storage
# - API/data pipeline
# - slow/fast layer architecture
# - strategy builder details
# - .env / environments
# - report/LaTeX
# - DevOps / CI/CD
# - testing / demo validation
# - V1.0 closure
#
# Also:
# - removes/deletes the "poc" label from the repo
# - does NOT use "poc" label on new issues
# - adds issues to existing GitHub Project
# - sets Roadmap, Percentage and Deadline fields
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

# ------------------------------------------------------------
# Run Python orchestration
# ------------------------------------------------------------

$PYTHON_BIN - <<'PY'
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

P0 = "□□□□□□□□□□ 0%"

NOW = "✅ Now"
NEXT = "🔜 Next"
LATER = "🗓️ Later"

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
            errors="replace"
        )
    else:
        result = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace"
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

def label_exists(name):
    labels = gh_json([
        "gh", "label", "list",
        "--repo", GH_REPO,
        "--limit", "300",
        "--json", "name"
    ]) or []

    return any(label["name"] == name for label in labels)

def create_label(name, color, description):
    if label_exists(name):
        print(f"Label already exists: {name}")
        return

    run([
        "gh", "label", "create", name,
        "--repo", GH_REPO,
        "--color", color,
        "--description", description
    ])

    print(f"Created label: {name}")

def delete_label_if_exists(name):
    if label_exists(name):
        print(f"Deleting label: {name}")
        run([
            "gh", "label", "delete", name,
            "--repo", GH_REPO,
            "--yes"
        ], check=False)
    else:
        print(f"Label not present: {name}")

def issue_find_exact(title):
    results = gh_json([
        "gh", "issue", "list",
        "--repo", GH_REPO,
        "--state", "all",
        "--search", f'"{title}" in:title',
        "--limit", "100",
        "--json", "number,title,url"
    ]) or []

    for issue in results:
        if issue.get("title") == title:
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

    # Fetch again so we have number/title/url reliably.
    created = issue_find_exact(task["title"])
    if not created:
        created = {
            "number": None,
            "title": task["title"],
            "url": url,
        }

    return created

def add_issue_to_project(issue_url):
    if not issue_url:
        return

    result = subprocess.run([
    "gh", "project", "item-add", PROJECT_NUMBER,
    "--owner", OWNER,
    "--url", issue_url
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    # It is okay if it already exists.
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
        "--format", "json"
    ])

    return data["id"]

def get_project_fields():
    data = gh_json([
        "gh", "project", "field-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "100",
        "--format", "json"
    ])

    return data["fields"]

def get_project_items():
    data = gh_json([
        "gh", "project", "item-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "300",
        "--format", "json"
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

def set_project_fields(project_id, item_id, roadmap_field_id, roadmap_option_id, percentage_field_id, percentage_option_id, deadline_field_id, deadline):
    query = """
    mutation(
      $projectId: ID!,
      $itemId: ID!,
      $roadmapFieldId: ID!,
      $roadmapOptionId: String!,
      $percentageFieldId: ID!,
      $percentageOptionId: String!,
      $deadlineFieldId: ID!,
      $deadline: Date!
    ) {
      roadmap: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $roadmapFieldId,
        value: { singleSelectOptionId: $roadmapOptionId }
      }) {
        projectV2Item { id }
      }

      percentage: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $percentageFieldId,
        value: { singleSelectOptionId: $percentageOptionId }
      }) {
        projectV2Item { id }
      }

      deadline: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $deadlineFieldId,
        value: { date: $deadline }
      }) {
        projectV2Item { id }
      }
    }
    """

    run([
        "gh", "api", "graphql",
        "-f", f"query={query}",
        "-f", f"projectId={project_id}",
        "-f", f"itemId={item_id}",
        "-f", f"roadmapFieldId={roadmap_field_id}",
        "-f", f"roadmapOptionId={roadmap_option_id}",
        "-f", f"percentageFieldId={percentage_field_id}",
        "-f", f"percentageOptionId={percentage_option_id}",
        "-f", f"deadlineFieldId={deadline_field_id}",
        "-f", f"deadline={deadline}",
    ])

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

    if rate["remaining"] < 250:
        print("")
        print("WARNING: GraphQL remaining is low.")
        print("This script can still run, but if it fails, wait until resetAt and run it again.")
        print("The script is idempotent and can safely be rerun.")

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

labels = [
    ("data", "C5DEF5", "Data pipeline, DuckDB, yfinance, features"),
    ("frontend", "BFD4F2", "Frontend, UI, charts, dashboard"),
    ("backend", "D4C5F9", "Backend, API, app logic"),
    ("rl", "FBCA04", "Reinforcement learning, FinRL, IQN"),
    ("finrl", "FBCA04", "FinRL data, environments, agents and backtests"),
    ("gymnasium", "FBCA04", "Gymnasium-compatible trading environments"),
    ("iqn", "FBCA04", "Implicit Quantile Networks and distributional RL"),
    ("uncertainty", "B60205", "Uncertainty estimation and confidence scoring"),
    ("risk", "B60205", "Risk, CVaR, drawdown, quantiles"),
    ("strategy", "F9D0C4", "Strategy builder, strategy configs and constraints"),
    ("decision", "D93F0B", "Decision engine and decision alternatives"),
    ("portfolio", "1D76DB", "Portfolio creation and state"),
    ("audit", "006B75", "Audit log, point-in-time, transparency"),
    ("duckdb", "C5DEF5", "DuckDB schema, queries and storage"),
    ("api", "BFD4F2", "External APIs and ingestion clients"),
    ("features", "C5DEF5", "Feature engineering and technical indicators"),
    ("macro", "C5DEF5", "Macro indicators such as inflation and rates"),
    ("fundamentals", "C5DEF5", "Company fundamentals and financial statements"),
    ("guldnas", "5319E7", "guldNAS Raspberry Pi persistent storage"),
    ("docker", "2496ED", "Dockerfiles, Docker Compose and container setup"),
    ("k3s", "5319E7", "k3s and Turing Pi deployment target"),
    ("infra", "5319E7", "Infrastructure, storage, deployment and runtime"),
    ("architecture", "5319E7", "Architecture and system design"),
    ("environment", "C2E0C6", ".env files and environment configuration"),
    ("local", "C2E0C6", "Local development environment"),
    ("test-env", "C2E0C6", "Test environment"),
    ("production", "C2E0C6", "Production environment"),
    ("report", "0E8A16", "Thesis report writing"),
    ("latex", "0E8A16", "LaTeX report template and build"),
    ("bibtex", "0E8A16", "BibTeX references"),
    ("documentation", "0075CA", "README, docs, runbooks"),
    ("evaluation", "5319E7", "Backtesting, metrics, results and evaluation"),
    ("test", "D4C5F9", "Manual or automated tests"),
    ("devops", "5319E7", "CI/CD, deployment workflows and automation"),
    ("ci", "5319E7", "GitHub Actions CI checks"),
    ("monitoring", "5319E7", "Monitoring and logging"),
    ("rollback", "5319E7", "Rollback strategy"),
    ("planning", "F9D0C4", "Planning, scope and Definition of Done"),
    ("worker", "D4C5F9", "Background workers and scheduled jobs"),
    ("training", "FBCA04", "Offline training jobs and model execution"),
    ("urgent", "B60205", "Must be done immediately"),
    ("high", "D93F0B", "Important for PoC"),
    ("medium", "FBCA04", "Useful but not blocking"),
    ("low", "C2E0C6", "Nice to have"),
    ("bonus", "7057FF", "Bonus if time allows"),
    ("future-work", "C5DEF5", "Future work / perspectives"),
]

print("")
print("Creating required labels...")
for name, color, description in labels:
    create_label(name, color, description)

print("")
print("Removing/deleting 'poc' labels because this is already a PoC board...")
delete_label_if_exists("poc")
delete_label_if_exists("scope:poc")

# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

tasks = [
    # --------------------------------------------------------
    # FinRL
    # --------------------------------------------------------
    {
        "title": "Install and verify FinRL environment",
        "milestone": "M1 — PoC Foundation",
        "labels": ["finrl", "rl", "environment", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Install and verify that FinRL can run in the project environment.

## Acceptance criteria
- FinRL dependencies are installed in the active Python environment
- a minimal FinRL import test works
- environment setup is documented
- known installation issues are written down
- the setup does not block the local PoC app

## Notes
This task verifies the RL framework foundation. It does not require full training yet."""
    },
    {
        "title": "Create FinRL baseline data pipeline",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["finrl", "data", "duckdb", "urgent"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Create a baseline FinRL-compatible data pipeline.

## Acceptance criteria
- selected tickers can be loaded
- daily price data can be prepared for FinRL-style usage
- data can be stored in DuckDB
- train/validation/test split is defined
- pipeline is reproducible from a script

## Suggested assets
AAPL, MSFT, NVDA, SPY."""
    },
    {
        "title": "Train FinRL baseline agent",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["finrl", "rl", "training", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Train one simple FinRL baseline agent.

## Acceptance criteria
- one standard agent can be trained or loaded
- training period is separated from test period
- model output/checkpoint path is defined
- training runtime and limitations are documented
- output can be used for report discussion

## Scope
This is a minimal baseline, not extensive hyperparameter tuning."""
    },
    {
        "title": "Create buy-and-hold and equal-weight baselines",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["evaluation", "rl", "risk", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Create simple non-RL baselines for comparison.

## Acceptance criteria
- buy-and-hold baseline implemented
- equal-weight baseline implemented
- metrics are comparable with RL output
- results can be exported for thesis tables

## Metrics
- cumulative return
- Sharpe ratio
- max drawdown
- CVaR proxy if available."""
    },
    {
        "title": "Export RL and backtest metrics to DuckDB",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["evaluation", "duckdb", "report", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Store model/backtest metrics in DuckDB so the thesis can use reproducible outputs.

## Acceptance criteria
- results table exists in DuckDB
- model name, strategy, dataset snapshot and metrics are stored
- outputs can be queried for report tables
- export script exists

## Thesis relevance
This supports reproducible results and report generation from current data."""
    },
    {
        "title": "Document FinRL limitations and assumptions",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["finrl", "report", "documentation", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Document what the PoC does and does not prove about FinRL.

## Acceptance criteria
- limitations of training time are documented
- limited hyperparameter tuning is justified
- baseline comparison is explained
- distinction between PoC and full production model is clear
- future improvements are listed."""
    },

    # --------------------------------------------------------
    # .env / environments
    # --------------------------------------------------------
    {
        "title": "Define .env.example for local PoC",
        "milestone": "M1 — PoC Foundation",
        "labels": ["environment", "local", "documentation", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Create the .env.example file required to run the PoC locally.

## Acceptance criteria
- .env.example exists
- app password / secret is defined
- DuckDB path is configurable
- data storage path is configurable
- environment name can be set
- no real secrets are committed."""
    },
    {
        "title": "Verify local app: front page and login",
        "milestone": "M1 — PoC Foundation",
        "labels": ["test", "frontend", "local", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Verify that the local PoC app starts and login works.

## Acceptance criteria
- app starts locally
- front page loads
- login/password gate works
- failed login gives sensible feedback
- successful login opens the dashboard."""
    },
    {
        "title": "Define environment configuration: local, test and production",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["environment", "infra", "documentation", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Define how local, test and production environments differ.

## Acceptance criteria
- local configuration is documented
- test/k3s configuration is documented
- production route target is documented
- environment variables are listed
- secrets handling is described at PoC level."""
    },
    {
        "title": "Verify test deployment on k3s",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["k3s", "test-env", "infra", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Verify whether the PoC can be deployed to the test k3s environment.

## Acceptance criteria
- namespace exists
- app container can be deployed or deployment path is documented
- service can be reached internally
- storage limitation is documented
- this task does not block V1.0 local PoC."""
    },
    {
        "title": "Verify production route on guldmand.com thesis path",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["production", "infra", "documentation", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Document and optionally verify the production route for the thesis demo.

## Target path
guldmand.com/data-science/master-thesis

## Acceptance criteria
- target route is documented
- reverse proxy / routing assumption is described
- authentication requirement is described
- if not implemented, it is clearly marked as future work."""
    },

    # --------------------------------------------------------
    # Report / LaTeX
    # --------------------------------------------------------
    {
        "title": "Define report.tex structure",
        "milestone": "M1 — PoC Foundation",
        "labels": ["report", "latex", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Create the main LaTeX report structure.

## Acceptance criteria
- report/report.tex exists
- sections are defined
- figures and tables paths are defined
- compilation command is documented
- empty sections can be filled incrementally."""
    },
    {
        "title": "Define BibTeX reference file",
        "milestone": "M1 — PoC Foundation",
        "labels": ["report", "bibtex", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Create the BibTeX reference file for the thesis.

## Acceptance criteria
- report/references.bib exists
- core RL, FinRL, IQN, CVaR and finance references are added
- citation keys are consistent
- report.tex can cite at least one paper."""
    },
    {
        "title": "Create thesis figures folder and export workflow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["report", "documentation", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Create a consistent workflow for saving thesis figures.

## Acceptance criteria
- report/figures exists
- report/tables exists
- screenshot naming convention is documented
- generated tables/plots can be exported from code
- README4 references the output locations."""
    },
    {
        "title": "Define results table format from PoC outputs",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["report", "evaluation", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Define the result table format used in the thesis.

## Acceptance criteria
- table columns are defined
- metrics are defined
- model/strategy/dataset identifiers are included
- one example table can be generated."""
    },
    {
        "title": "Create reproducible report build command",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["report", "latex", "devops", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Make the thesis report build reproducibly.

## Acceptance criteria
- build command is documented
- LaTeX artifacts are ignored by Git
- report can be built from the report folder
- common build errors are documented if needed."""
    },

    # --------------------------------------------------------
    # DevOps / CI / operations
    # --------------------------------------------------------
    {
        "title": "Create GitHub Actions CI workflow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["devops", "ci", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Create a minimal CI workflow for the repository.

## Acceptance criteria
- .github/workflows/ci.yml exists
- workflow runs on push/PR
- Python environment installs
- basic tests or import checks run
- workflow result is visible in GitHub."""
    },
    {
        "title": "Add Python lint and test workflow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["devops", "ci", "test", "medium"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Add a simple lint/test workflow for Python code.

## Acceptance criteria
- test command is defined
- lint or formatting check is defined
- failing tests fail the workflow
- workflow is lightweight enough for rapid iteration."""
    },
    {
        "title": "Add Docker build workflow",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["devops", "docker", "ci", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Add a workflow that verifies the Docker image builds.

## Acceptance criteria
- Docker build workflow exists
- Dockerfile is built in CI
- failure blocks merge or is clearly visible
- workflow does not deploy anything automatically yet."""
    },
    {
        "title": "Add Dependabot configuration",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "documentation", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Add Dependabot configuration for dependency visibility.

## Acceptance criteria
- .github/dependabot.yml exists
- Python dependencies are covered
- GitHub Actions dependencies are covered
- update frequency is documented."""
    },
    {
        "title": "Define deployment workflow for test environment",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "k3s", "test-env", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Define how the app should be deployed to the test k3s environment.

## Acceptance criteria
- deployment steps are documented
- image build/push assumptions are documented
- environment variables are documented
- manual deployment command is described
- automation is marked as v1.1/future if not implemented."""
    },
    {
        "title": "Define production deployment and rollback strategy",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "production", "rollback", "low"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Document production deployment and rollback strategy.

## Acceptance criteria
- production deployment concept is described
- rollback condition is described
- rollback command/approach is outlined
- monitoring requirement is documented
- marked as future work if not implemented."""
    },
    {
        "title": "Define basic logging strategy",
        "milestone": "M3 — Decision Support",
        "labels": ["monitoring", "backend", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-12",
        "body": """## Goal
Define basic app logging for the PoC.

## Acceptance criteria
- log format is chosen
- important events are listed
- errors are logged
- decision/audit logs are kept separate from technical logs."""
    },
    {
        "title": "Define audit log export for thesis demo",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["audit", "report", "evaluation", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Export audit logs for the thesis demo.

## Acceptance criteria
- decision logs can be exported
- export contains strategy, portfolio, market snapshot and user choice
- export can be used as thesis evidence
- output path is documented."""
    },
    {
        "title": "Define monitoring strategy for future deployment",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["monitoring", "future-work", "low"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Document how monitoring should work in a later deployment.

## Acceptance criteria
- health checks are described
- logs/metrics idea is described
- model/data drift monitoring is mentioned
- marked as future work."""
    },
    {
        "title": "Define rollback strategy for failed deployment",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["rollback", "devops", "future-work", "low"],
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Document rollback behavior for failed deployment.

## Acceptance criteria
- rollback trigger is defined
- previous working image/version concept is described
- database/data safety considerations are noted
- marked as future work."""
    },

    # --------------------------------------------------------
    # Gymnasium environment
    # --------------------------------------------------------
    {
        "title": "Define Gymnasium trading environment interface",
        "milestone": "M3 — Decision Support",
        "labels": ["gymnasium", "rl", "architecture", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-12",
        "body": """## Goal
Define the Gymnasium-compatible environment interface for the trading setup.

## Acceptance criteria
- reset() and step() behavior is described
- observation/state shape is described
- action representation is described
- reward components are described
- long-only constraint is included

## Scope
This task defines the interface before full implementation."""
    },
    {
        "title": "Implement long-only Gymnasium environment prototype",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["gymnasium", "rl", "backend", "urgent"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Implement a minimal long-only trading environment.

## Acceptance criteria
- environment can reset
- environment can step through daily market data
- action space excludes shorting
- cash and positions update correctly
- transaction costs are represented
- environment can run with random policy."""
    },
    {
        "title": "Define state, action and reward schema",
        "milestone": "M3 — Decision Support",
        "labels": ["rl", "strategy", "decision", "urgent"],
        "roadmap": NEXT,
        "deadline": "2026-05-12",
        "body": """## Goal
Define the core RL schema used by the PoC.

## Acceptance criteria
- state fields are listed
- action alternatives are listed
- reward formula is written
- transaction cost penalty is included
- risk penalty is included
- strategy constraints are represented

## Important note
Strategy switching should be treated as a higher-level decision alternative unless explicitly implemented as an environment action."""
    },
    {
        "title": "Validate environment with random policy",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["gymnasium", "rl", "evaluation", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Validate that the environment behaves sensibly using a random policy.

## Acceptance criteria
- random policy can run without crashing
- portfolio values update
- impossible actions are handled
- metrics can be computed
- output is documented."""
    },

    # --------------------------------------------------------
    # guldNAS / persistent data
    # --------------------------------------------------------
    {
        "title": "Create guldNAS storage folder structure",
        "milestone": "M1 — PoC Foundation",
        "labels": ["guldnas", "infra", "data", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Create the intended persistent storage structure on guldNAS.

## Acceptance criteria
- root folder is defined
- DuckDB folder is defined
- parquet/raw/curated/features folders are defined
- model-checkpoints folder is defined
- logs/results folders are defined
- structure is documented."""
    },
    {
        "title": "Mount guldNAS storage path for local development",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["guldnas", "infra", "data", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Make guldNAS available from the development machine.

## Acceptance criteria
- mount path is documented
- local app can reference storage path
- fallback local runtime-data path is defined
- permissions assumptions are documented."""
    },
    {
        "title": "Define DuckDB canonical path on guldNAS",
        "milestone": "M1 — PoC Foundation",
        "labels": ["duckdb", "guldnas", "data", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Define where the canonical DuckDB file should live.

## Acceptance criteria
- canonical path is documented
- local fallback path is documented
- .env variable name is defined
- backup/snapshot idea is noted

## Example
GULDNAS_DUCKDB_PATH=/mnt/guldNAS/stockinvestmentdss/duckdb/market_research.duckdb"""
    },
    {
        "title": "Define Parquet raw curated features layout",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["data", "features", "guldnas", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Define the file-based data layout for raw, curated and feature data.

## Acceptance criteria
- raw folder layout is defined
- curated folder layout is defined
- features folder layout is defined
- naming convention includes source/ticker/date where relevant
- relation to DuckDB is documented."""
    },
    {
        "title": "Document storage backup and snapshot assumptions",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["documentation", "guldnas", "infra", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Document how backups and snapshots should work for storage.

## Acceptance criteria
- what must be backed up is listed
- snapshot idea is described
- thesis reproducibility implications are described
- limitations are stated."""
    },

    # --------------------------------------------------------
    # API / data pipeline
    # --------------------------------------------------------
    {
        "title": "Define point-in-time ingestion schema",
        "milestone": "M1 — PoC Foundation",
        "labels": ["data", "audit", "duckdb", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Define the schema needed for point-in-time data handling.

## Acceptance criteria
- event_time is defined
- ingestion_time is defined
- source is defined
- source_version or source_request_id is defined
- dataset snapshot/build id concept is described

## Thesis relevance
This supports transparency, auditability and avoiding look-ahead bias."""
    },
    {
        "title": "Implement raw file storage for API responses",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["api", "data", "guldnas", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Store raw API responses before transforming them.

## Acceptance criteria
- raw output folder exists
- raw responses are saved with timestamp/source/ticker
- file naming convention is documented
- raw files can later be ingested into DuckDB."""
    },
    {
        "title": "Implement DuckDB ingestion from raw files",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["duckdb", "data", "api", "urgent"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Load raw API data into DuckDB.

## Acceptance criteria
- raw files can be parsed
- rows are inserted/upserted into DuckDB
- duplicates are avoided
- event_time and ingestion_time are stored
- ingestion script is documented."""
    },
    {
        "title": "Create market data ingestion pipeline",
        "milestone": "M1 — PoC Foundation",
        "labels": ["data", "api", "duckdb", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Create the first working market data ingestion pipeline.

## Acceptance criteria
- selected tickers can be fetched
- data is saved as raw file
- data is stored in DuckDB
- pipeline can be rerun safely
- pipeline supports the demo tickers."""
    },
    {
        "title": "Create technical indicator feature pipeline",
        "milestone": "M3 — Decision Support",
        "labels": ["features", "data", "risk", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Create basic technical indicators for decision support.

## Acceptance criteria
- returns are calculated
- rolling volatility is calculated
- moving average or RSI/MACD placeholder exists
- features are stored in DuckDB or Parquet
- features can be used by decision cards."""
    },
    {
        "title": "Create macro indicator ingestion placeholder",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["macro", "data", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Create a placeholder for macro indicator ingestion.

## Acceptance criteria
- inflation/rate indicator target is documented
- schema placeholder exists
- integration is marked as PoC placeholder if not fully implemented
- thesis explanation is clear."""
    },
    {
        "title": "Create fundamentals ingestion placeholder",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["fundamentals", "data", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Create a placeholder for company fundamentals ingestion.

## Acceptance criteria
- candidate fields are listed
- schema placeholder exists
- point-in-time concern is documented
- integration is marked as PoC placeholder if not fully implemented."""
    },
    {
        "title": "Create dataset build ID and snapshot logging",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["audit", "data", "duckdb", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Track dataset builds and snapshots.

## Acceptance criteria
- dataset_build_id is defined
- snapshot metadata table exists or is planned
- model outputs can reference dataset snapshot
- decision logs can reference market snapshot."""
    },

    # --------------------------------------------------------
    # IQN / uncertainty
    # --------------------------------------------------------
    {
        "title": "Implement IQN-style quantile output",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["iqn", "risk", "rl", "urgent"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Expose quantile-style output for decision support.

## Acceptance criteria
- q10 is shown
- q50 is shown
- q90 is shown
- downside risk is shown
- output appears on decision cards
- implementation is clearly marked as IQN-style/proxy if not full IQN."""
    },
    {
        "title": "Implement evidential uncertainty proxy",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["uncertainty", "risk", "rl", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Implement a simple uncertainty proxy for PoC decision support.

## Acceptance criteria
- uncertainty score is calculated
- uncertainty label low/medium/high is shown
- explanation is included
- limitations are documented
- full evidential model can be future work if needed."""
    },
    {
        "title": "Document full IQN and evidential model as extension if needed",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["report", "iqn", "uncertainty", "future-work"],
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Document the full intended IQN/evidential setup if the PoC only implements proxies.

## Acceptance criteria
- full intended method is described
- PoC limitation is stated honestly
- future work implementation path is outlined
- relation to decision support is clear."""
    },

    # --------------------------------------------------------
    # Slow / fast layer
    # --------------------------------------------------------
    {
        "title": "Define slow-layer offline training architecture",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["architecture", "rl", "report", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Define the offline training layer.

## Acceptance criteria
- training data flow is described
- model checkpoint path is described
- model registry idea is described
- GPU/cloud training target is described
- not used for live user interaction."""
    },
    {
        "title": "Define fast-layer online decision support architecture",
        "milestone": "M1 — PoC Foundation",
        "labels": ["architecture", "decision", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Define the fast decision support layer.

## Acceptance criteria
- no live retraining requirement
- current portfolio state is used
- latest/cached features are used
- user constraints are applied
- decision alternatives are generated immediately."""
    },
    {
        "title": "Create model registry schema in DuckDB",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["duckdb", "rl", "audit", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Create or define a model registry schema.

## Acceptance criteria
- model_id is defined
- strategy/profile relation is stored
- dataset snapshot relation is stored
- checkpoint path is stored
- evaluation metrics can be linked."""
    },
    {
        "title": "Create pretrained model selection logic",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["rl", "decision", "strategy", "medium"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Select the nearest pretrained model/profile for a user strategy.

## Acceptance criteria
- custom strategy can map to profile
- fallback model logic is defined
- missing model case is handled
- limitations are documented."""
    },
    {
        "title": "Document slow fast split in report",
        "milestone": "M3 — Decision Support",
        "labels": ["report", "architecture", "urgent"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Write the slow/fast layer explanation in the report.

## Acceptance criteria
- offline training is explained
- online inference/decision support is explained
- no live retraining per user decision is stated
- architecture diagram references this split."""
    },

    # --------------------------------------------------------
    # Strategy builder details
    # --------------------------------------------------------
    {
        "title": "Define strategy JSON schema",
        "milestone": "M1 — PoC Foundation",
        "labels": ["strategy", "backend", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Define the structured strategy schema used by predefined and custom strategies.

## Acceptance criteria
- risk_profile is included
- objective is included
- max_position_size is included
- cash_buffer is included
- stop-loss and take-profit fields are included
- rebalance_frequency is included
- locked_assets is included
- long_only is explicit."""
    },
    {
        "title": "Implement custom strategy form fields",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["strategy", "frontend", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Implement the form fields for custom strategy creation.

## Acceptance criteria
- risk slider is present
- objective can be selected
- max position size can be set
- cash buffer can be set
- stop-loss/take-profit can be configured
- strategy JSON is saved."""
    },
    {
        "title": "Implement locked assets constraint",
        "milestone": "M3 — Decision Support",
        "labels": ["strategy", "decision", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Allow the user to lock assets they prefer not to sell.

## Acceptance criteria
- locked asset can be stored in strategy/portfolio state
- decision engine respects locked assets
- system can still suggest risk-reducing alternatives
- explanation is shown to the user."""
    },
    {
        "title": "Implement strategy switch decision alternative",
        "milestone": "M3 — Decision Support",
        "labels": ["strategy", "decision", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Generate strategy switching as a decision alternative.

## Acceptance criteria
- current strategy is compared with portfolio/market state
- alternative strategy is suggested when relevant
- explanation includes trade-offs
- user remains final decision-maker."""
    },
    {
        "title": "Connect strategy config to decision engine",
        "milestone": "M3 — Decision Support",
        "labels": ["strategy", "backend", "decision", "urgent"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Use the saved strategy configuration inside the decision engine.

## Acceptance criteria
- decision engine reads strategy config
- constraints affect generated alternatives
- stop-loss/take-profit rules affect decision cards
- risk profile affects explanations."""
    },

    # --------------------------------------------------------
    # Test/demo flows
    # --------------------------------------------------------
    {
        "title": "Test local login flow",
        "milestone": "M1 — PoC Foundation",
        "labels": ["test", "frontend", "local", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Test login flow in the local PoC app.

## Acceptance criteria
- login page opens
- wrong password fails
- correct password succeeds
- dashboard opens after login."""
    },
    {
        "title": "Test user creation flow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["test", "frontend", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Test investor/user creation.

## Acceptance criteria
- user/investor name can be entered
- user is saved
- user can be loaded again
- user appears in demo flow."""
    },
    {
        "title": "Test stock lookup flow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["test", "data", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Test stock lookup in the app.

## Acceptance criteria
- ticker can be entered
- market data is fetched or loaded
- result is shown to the user
- missing ticker case is handled."""
    },
    {
        "title": "Test portfolio creation flow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["test", "portfolio", "high"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Test portfolio creation.

## Acceptance criteria
- initial capital can be entered
- positions can be added
- buy price/date can be entered
- current value and gain/loss are shown."""
    },
    {
        "title": "Test strategy builder flow",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["test", "strategy", "urgent"],
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Test the guided strategy builder.

## Acceptance criteria
- predefined strategy can be selected
- custom strategy form opens
- custom strategy can be saved
- saved strategy affects later decision support."""
    },
    {
        "title": "Test decision card generation",
        "milestone": "M3 — Decision Support",
        "labels": ["test", "decision", "urgent"],
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Test that decision alternatives are generated.

## Acceptance criteria
- hold option appears
- reduce/sell/rebalance option appears when relevant
- explanation appears
- risk/uncertainty fields appear or placeholders are shown."""
    },
    {
        "title": "Test audit log creation",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["test", "audit", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-13",
        "body": """## Goal
Test that decisions are stored in the audit log.

## Acceptance criteria
- user choice is saved
- strategy state is saved
- portfolio state is saved
- market snapshot reference is saved
- decision can be inspected later."""
    },
    {
        "title": "Test full NVDA demo case",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["test", "evaluation", "report", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Run the full end-to-end NVDA-style demo case.

## Acceptance criteria
- user logs in
- portfolio is created
- strategy is selected/created
- NVDA-style position is evaluated
- hold/reduce/sell/switch alternatives are shown
- audit log is saved
- screenshots/results can be used in thesis."""
    },

    # --------------------------------------------------------
    # V1.0 closure
    # --------------------------------------------------------
    {
        "title": "Define V1.0 Definition of Done",
        "milestone": "M1 — PoC Foundation",
        "labels": ["planning", "documentation", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Define exactly when V1.0 is considered done.

## Acceptance criteria
- login works locally
- user/investor can be created
- strategy can be selected or created
- portfolio can be created
- stock data can be fetched
- decision alternatives are shown
- q10/q50/q90 or proxy uncertainty is shown
- decision is saved in audit log
- README4 explains how to run it

## Stop rule
Anything not needed for this definition is v1.1 or future work."""
    },
    {
        "title": "Create final demo script",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["documentation", "test", "report", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Create a step-by-step script for demonstrating the PoC.

## Acceptance criteria
- demo steps are written
- fixed demo portfolio is defined
- fixed demo strategy is defined
- NVDA-style sell/hold/reduce case is included
- audit log is shown at the end."""
    },
    {
        "title": "Create thesis evidence package",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["report", "evaluation", "documentation", "high"],
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Collect the evidence needed for the thesis report.

## Acceptance criteria
- screenshots are saved
- result tables are exported
- demo data snapshot is saved
- strategy JSON is saved
- decision log export is saved
- architecture diagram is saved."""
    },
    {
        "title": "Freeze V1.0 scope",
        "milestone": "M1 — PoC Foundation",
        "labels": ["planning", "documentation", "urgent"],
        "roadmap": NOW,
        "deadline": "2026-05-10",
        "body": """## Goal
Freeze the V1.0 PoC scope.

## Acceptance criteria
- V1.0 scope is written in README4
- all non-essential work is marked v1.1 or future work
- no new feature issues are added to Now unless they unblock V1.0
- board reflects the five-day sprint plan."""
    },
]

# ------------------------------------------------------------
# Create/update issues
# ------------------------------------------------------------

check_graphql_rate_limit()

print("")
print("Creating or finding issues and adding them to project...")

for task in tasks:
    issue = create_or_get_issue(task)
    add_issue_to_project(issue.get("url"))
    time.sleep(0.15)

# Give GitHub Projects a moment to reflect item additions.
time.sleep(2)

# ------------------------------------------------------------
# Project fields
# ------------------------------------------------------------

print("")
print("Fetching project metadata...")

project_id = get_project_id()
fields = get_project_fields()

roadmap_field_id = field_id(fields, "Roadmap")
percentage_field_id = field_id(fields, "Percentage")
deadline_field_id = field_id(fields, "Deadline")

if not roadmap_field_id:
    print("ERROR: Project field not found: Roadmap")
    sys.exit(1)

if not percentage_field_id:
    print("ERROR: Project field not found: Percentage")
    sys.exit(1)

if not deadline_field_id:
    print("ERROR: Project field not found: Deadline")
    sys.exit(1)

roadmap_options = {
    NOW: option_id(fields, "Roadmap", NOW),
    NEXT: option_id(fields, "Roadmap", NEXT),
    LATER: option_id(fields, "Roadmap", LATER),
}

percentage_0_option_id = option_id(fields, "Percentage", P0)

for name, oid in roadmap_options.items():
    if not oid:
        print(f"ERROR: Roadmap option not found: {name}")
        sys.exit(1)

if not percentage_0_option_id:
    print("")
    print(f"ERROR: Percentage option not found: {P0}")
    print("Add this exact option manually to the Percentage field and rerun script 6.")
    sys.exit(1)

print("")
print("Fetching project items...")
items = get_project_items()

print("")
print("Setting Roadmap, Percentage and Deadline for new/missing tasks...")

for task in tasks:
    item_id = item_id_by_title(items, task["title"])

    if not item_id:
        print(f"WARNING: Project item not found for: {task['title']}")
        continue

    roadmap_option_id = roadmap_options[task["roadmap"]]

    print(f"Updating fields: {task['title']}")
    set_project_fields(
        project_id=project_id,
        item_id=item_id,
        roadmap_field_id=roadmap_field_id,
        roadmap_option_id=roadmap_option_id,
        percentage_field_id=percentage_field_id,
        percentage_option_id=percentage_0_option_id,
        deadline_field_id=deadline_field_id,
        deadline=task["deadline"],
    )

    print(f"  Roadmap:   {task['roadmap']}")
    print(f"  Percentage:{P0}")
    print(f"  Deadline:  {task['deadline']}")

    time.sleep(0.15)

print("")
print("Deleting 'poc' labels one more time in case anything recreated them...")
delete_label_if_exists("poc")
delete_label_if_exists("scope:poc")

print("")
print("Done.")
print("")
print("Script 6 completed.")
print("")
print("What was added:")
print("- FinRL tasks")
print("- Gymnasium environment tasks")
print("- guldNAS persistent storage tasks")
print("- API/data pipeline tasks")
print("- slow/fast layer tasks")
print("- strategy builder detail tasks")
print("- environment/.env tasks")
print("- report/LaTeX tasks")
print("- DevOps/CI tasks")
print("- testing/demo tasks")
print("- V1.0 closure tasks")
print("")
print("The 'poc' label has been removed/deleted.")
print("All created/found tasks have Roadmap, Percentage=0%, and Deadline set.")
PY