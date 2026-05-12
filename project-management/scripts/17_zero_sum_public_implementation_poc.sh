#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 17
# zero-sum-public implementation in PoC
# ============================================================
#
# Purpose:
#   Creates the next PoC issues for using zero-sum-public as a
#   frontend/product reference while keeping the current stack:
#   FastAPI backend, Jinja2 frontend, DuckDB runtime data,
#   Docker Compose local app and FinRL/yfinance-compatible data.
#
# Safe to re-run:
#   - Existing labels are reused.
#   - Existing issues with identical title are reused.
#   - Existing project items are reused.
#   - Project fields are updated idempotently.
#
# Usage from repository root on Windows PowerShell:
#   & "C:\Program Files\Git\bin\bash.exe" ".\project-management\scripts\17_zero_sum_public_implementation_poc.sh"
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

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
  echo ""
  echo "Login first:"
  echo "  gh auth login --web"
  echo ""
  echo "If project updates fail, refresh project scope:"
  echo "  gh auth refresh -s project"
  exit 1
}

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

echo ""
echo "Creating required labels..."

create_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label list --repo "$GH_REPO" --limit 500 --json name --jq '.[].name' | grep -Fxq "$name"; then
    echo "Label already exists: $name"
  else
    gh label create "$name" \
      --repo "$GH_REPO" \
      --color "$color" \
      --description "$description" >/dev/null
    echo "Created label: $name"
  fi
}

create_label "zero-sum-public" "1D76DB" "Zero Sum Public reference and implementation mapping"
create_label "frontend-reference" "BFDADC" "Frontend reference, UX inspiration and adaptation work"
create_label "market-data" "0E8A16" "Market data ingestion, storage and API"
create_label "portfolio" "C5DEF5" "Portfolio, holdings and transaction tracking"
create_label "watchlist" "D4C5F9" "Watchlist and saved ticker functionality"
create_label "stock-lookup" "5319E7" "Ticker search, stock pages and stock metadata"
create_label "analytics" "006B75" "Comparison, correlation and market analytics"
create_label "finrl" "FBCA04" "FinRL framework integration"
create_label "yfinance" "F9D0C4" "Yahoo Finance/yfinance data integration"
create_label "duckdb" "0E8A16" "DuckDB runtime and analytical storage"
create_label "application" "C5DEF5" "Application-facing PoC work"
create_label "backend" "5319E7" "Backend API and service work"
create_label "frontend" "BFD4F2" "Frontend UI, templates, styling or client-side behavior"
create_label "research" "0E8A16" "Research notebooks, experiments and thesis evidence"
create_label "architecture" "7057FF" "Architecture and system design"
create_label "documentation" "0075CA" "README, docs, runbooks and project notes"
create_label "fast-layer" "BFD4F2" "Online decision support and inference"
create_label "slow-layer" "5319E7" "Offline training, evaluation and scheduled jobs"
create_label "poc" "FBCA04" "Proof-of-concept scoped work"

echo ""
echo "Checking GraphQL rate limit before Project updates..."
gh api graphql -f query='{ rateLimit { limit cost remaining used resetAt } }' \
  --jq '.data.rateLimit | "  remaining: \(.remaining)\n  used:      \(.used)\n  resetAt:   \(.resetAt)"' || true

echo ""
echo "Creating/finding issues and updating project fields..."

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PYTHON_BIN" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

STATUS_TODO = "Todo"
PERCENT_0 = "□□□□□□□□□□ 0%"

CATEGORY_ARCHITECTURE = "🏗️ Architecture"
CATEGORY_DEVELOPMENT = "⚙️ Development"
CATEGORY_DATA = "📊 Data"
CATEGORY_EVALUATION = "🧪 Evaluation"

PRIORITY_URGENT = "🗼 Urgent"
PRIORITY_HIGH = "⛰️ High"
PRIORITY_MEDIUM = "🫣 Medium"
PRIORITY_LOW = "🌈 Low"

ROADMAP_NOW = "✅ Now"
ROADMAP_NEXT = "🔜 Next"
ROADMAP_LATER = "🗓️ Later"

TRACK_POC = "PoC"
TRACK_FUTURE = "Future Work"

MILESTONE_M2 = "M2 — Strategy + Portfolio"
MILESTONE_M3 = "M3 — Decision Support"

COMMON_SYSTEM_CONTEXT = """- External repos: `zero-sum-public` is reference only. Do not vendor the repo into the PoC unless a later issue explicitly decides it.
- Data pipeline: frontend must call backend API endpoints, not external APIs or DuckDB directly.
- Storage / guldNAS / DuckDB: local runtime DuckDB remains the first target; canonical guldNAS paths must remain configurable.
- Devices / infrastructure: local Docker Compose first; k3s/Turing Pi and GPU/cloud execution later.
- Containers / Docker / k3s: implementation must keep the local Compose app runnable.
- Research / Application split: research notebooks and FinRL experiments belong under `research/`; user-facing app code belongs under `system/`."""


def run(cmd: List[str], *, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        print("")
        print("ERROR running command:")
        print(" ".join(cmd))
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return (result.stdout or "").strip()


def gh_json(cmd: List[str]) -> Any:
    out = run(cmd)
    return json.loads(out) if out else None


def fetch_all_issues() -> List[Dict[str, Any]]:
    data = gh_json([
        "gh", "api",
        "--paginate",
        "--slurp",
        f"repos/{GH_REPO}/issues?state=all&per_page=100",
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
            })
    return issues


def issue_find_exact(title: str) -> Optional[Dict[str, Any]]:
    for issue in fetch_all_issues():
        if issue["title"] == title:
            return issue
    return None


def create_or_get_issue(task: Dict[str, Any]) -> Dict[str, Any]:
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
    ]

    if task.get("milestone"):
        cmd.extend(["--milestone", task["milestone"]])

    url = run(cmd)
    created = issue_find_exact(task["title"])
    if created:
        return created
    return {"number": None, "title": task["title"], "url": url}


def add_issue_to_project(issue_url: str) -> None:
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


def get_project_id() -> Optional[str]:
    data = gh_json([
        "gh", "project", "view", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
    ])
    return data.get("id") if data else None


def get_project_fields() -> List[Dict[str, Any]]:
    data = gh_json([
        "gh", "project", "field-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "100",
        "--format", "json",
    ])
    return data.get("fields", []) if data else []


def get_project_items() -> List[Dict[str, Any]]:
    data = gh_json([
        "gh", "project", "item-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--limit", "300",
        "--format", "json",
    ])
    return data.get("items", []) if data else []


def find_project_item_id(issue_title: str) -> Optional[str]:
    for item in get_project_items():
        content = item.get("content") or {}
        if content.get("title") == issue_title:
            return item.get("id")
    return None


def find_field(fields: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for field in fields:
        if field.get("name") == name:
            return field
    return None


def option_id(field: Dict[str, Any], option_name: str) -> Optional[str]:
    for option in field.get("options", []):
        if option.get("name") == option_name:
            return option.get("id")
    return None


def update_single_select(project_id: str, item_id: str, fields: List[Dict[str, Any]], field_name: str, option_name: str) -> None:
    field = find_field(fields, field_name)
    if not field:
        print(f"  Field missing, skipped: {field_name}")
        return
    opt_id = option_id(field, option_name)
    if not opt_id:
        print(f"  Option missing, skipped: {field_name}={option_name}")
        return
    run([
        "gh", "project", "item-edit",
        "--project-id", project_id,
        "--id", item_id,
        "--field-id", field["id"],
        "--single-select-option-id", opt_id,
    ])
    print(f"  {field_name}: {option_name}")


def update_number(project_id: str, item_id: str, fields: List[Dict[str, Any]], field_name: str, value: int) -> None:
    field = find_field(fields, field_name)
    if not field:
        print(f"  Field missing, skipped: {field_name}")
        return
    run([
        "gh", "project", "item-edit",
        "--project-id", project_id,
        "--id", item_id,
        "--field-id", field["id"],
        "--number", str(value),
    ])
    print(f"  {field_name}: {value}")


def update_date(project_id: str, item_id: str, fields: List[Dict[str, Any]], field_name: str, value: str) -> None:
    field = find_field(fields, field_name)
    if not field:
        print(f"  Field missing, skipped: {field_name}")
        return
    run([
        "gh", "project", "item-edit",
        "--project-id", project_id,
        "--id", item_id,
        "--field-id", field["id"],
        "--date", value,
    ])
    print(f"  {field_name}: {value}")


def body(goal: str, description: str, track: str, layer: str, system_context: str, implementation: str, test: str, acceptance: str, notes: str) -> str:
    return f"""## Goal

{goal.strip()}

## Description

{description.strip()}

## Track

{track.strip()}

## Layer

{layer.strip()}

## System Context

{system_context.strip()}

## Implementation

{implementation.strip()}

## Test

{test.strip()}

## Acceptance Criteria

{acceptance.strip()}

## Notes

{notes.strip()}
""".strip() + "\n"

TASKS: List[Dict[str, Any]] = [
    {
        "title": "Map zero-sum-public features to StockInvestmentDSS PoC",
        "milestone": MILESTONE_M2,
        "labels": ["zero-sum-public", "frontend-reference", "architecture", "documentation", "application", "poc"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "progress": 0,
        "body": body(
            "Map the useful Zero Sum Public features to the StockInvestmentDSS PoC before implementing more frontend pages.",
            """The PoC now has a working local app, Jinja2 frontend, backend API, login/session guard and Docker Compose flow.

Before building portfolio, stock lookup, watchlist, comparison and market overview features from scratch, review `zero-sum-public` as a reference and decide which ideas should be adapted into the current stack.

Relevant reference routes include:

```text
/portfolio
/stocks/AAPL
/watchlist
/compare?tickers=AAPL,MSFT
/correlation
/chart
/technical
/scanner
/heatmap
/bubble
/sectors
/news?symbol=AAPL
/earnings
/learn
```

The output should be a short mapping document, not a rewrite or framework migration.""",
            """- Application track: yes, because this defines user-facing PoC pages.
- Research track: yes, because later model outputs and risk/RL evidence need application surfaces.
- Shared track: yes, because it affects frontend, backend APIs, DuckDB and report evidence.""",
            """- Fast layer: yes, because mapped pages should support online decision support.
- Slow layer: indirect, because slow-layer FinRL/model outputs may later feed these pages.""",
            COMMON_SYSTEM_CONTEXT,
            """Create:

```text
system/docs/zero-sum-public-reference-map.md
```

Include:

```text
1. Reference pages reviewed
2. What to adapt
3. What to defer
4. Current-stack implementation target
5. Required backend endpoints
6. Required DuckDB tables
7. Required frontend templates/views
8. FinRL/yfinance/data dependency notes
9. Priority order
```

Recommended first priority:

```text
1. Market data foundation
2. Stock lookup/detail
3. Portfolio/watchlist/transactions
4. Compare/correlation
5. Market overview placeholders
```""",
            """- Verify the document exists.
- Verify it clearly states reference-only use of `zero-sum-public`.
- Verify it does not propose Next.js/React migration.
- Verify it identifies first implementation issue.
- Verify Docker Compose app still starts after any documentation changes.""",
            """- Mapping document exists.
- Zero Sum reference pages are listed.
- Features to adapt are listed.
- Deferred features are listed.
- Frontend page targets are listed.
- Backend API targets are listed.
- DuckDB table needs are listed.
- FinRL/yfinance relationship is noted.
- The implementation order is clear.""",
            "This issue prevents us from building 63 weak placeholders. It should be concise but decisive.",
        ),
    },
    {
        "title": "Define zero-sum-public adaptation boundary",
        "milestone": MILESTONE_M2,
        "labels": ["zero-sum-public", "frontend-reference", "architecture", "documentation", "poc"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "progress": 0,
        "body": body(
            "Define exactly how Zero Sum Public may be used in the PoC without destabilizing the current architecture.",
            """The project can benefit from Zero Sum's portfolio, stock, chart, watchlist and market analytics ideas, but the current PoC should not become a Next.js migration project.

This issue defines the boundary:

```text
Allowed:
- visual reference
- UX flow reference
- feature decomposition
- chart/page ideas
- naming inspiration
- lightweight algorithmic ideas if reimplemented cleanly

Not allowed without explicit later decision:
- vendoring the full repo
- copying large app structure blindly
- switching framework
- bypassing backend API
- frontend directly reading DuckDB
```""",
            """- Application track: yes, because it protects the frontend/app direction.
- Research track: indirect, because application surfaces must later display research outputs.
- Shared track: yes, because boundaries affect architecture, data and documentation.""",
            """- Fast layer: yes, because Zero Sum-inspired features are fast-layer user views.
- Slow layer: indirect, because slow-layer outputs must remain separate.""",
            COMMON_SYSTEM_CONTEXT,
            """Create or update:

```text
system/docs/zero-sum-public-reference-map.md
```

Add a section:

```text
## Adaptation Boundary
```

Define:

- what may be reused conceptually
- what must be reimplemented in current stack
- what must not be copied
- how to cite/reference the repo in internal docs
- when to create a separate issue for deeper extraction""",
            """- Verify the boundary is written down.
- Verify current stack remains FastAPI + Jinja2 + DuckDB.
- Verify frontend calls backend API only.
- Verify no external repo code is copied as part of this task.""",
            """- Adaptation boundary exists.
- Allowed use is clear.
- Forbidden use is clear.
- Current-stack strategy is preserved.
- No framework migration is introduced.
- No external code is vendored.""",
            "This is a guardrail issue. It should be completed before large Zero Sum-inspired implementation tasks.",
        ),
    },
    {
        "title": "Create market data foundation with FinRL-yfinance and DuckDB",
        "milestone": MILESTONE_M2,
        "labels": ["market-data", "finrl", "yfinance", "duckdb", "backend", "research", "application", "poc"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_URGENT,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_POC,
        "deadline": "2026-05-13",
        "progress": 0,
        "body": body(
            "Create the first real market data foundation for stock lookup, portfolio and later FinRL experiments.",
            """The application needs real market data before portfolio, watchlist, strategy and recommendations can become meaningful.

This task creates a minimal market data foundation that can fetch or load daily OHLCV data, persist it in DuckDB and expose it through backend API endpoints.

FinRL is required for the thesis direction, but this issue should keep V1.0 pragmatic: yfinance-compatible data ingestion is acceptable as the first fast implementation, as long as the structure remains compatible with later FinRL workflows.""",
            """- Application track: yes, because stock/portfolio pages need market data.
- Research track: yes, because the same data foundation should support FinRL, ObjectRL/Gymnasium and thesis experiments.
- Shared track: yes, because backend, DuckDB and frontend depend on the same data.""",
            """- Fast layer: yes, because backend APIs should serve stored market data to the app.
- Slow layer: indirect, because ingestion and feature generation can later become scheduled slow-layer jobs.""",
            COMMON_SYSTEM_CONTEXT,
            """Suggested backend files:

```text
system/backend/app/market_data.py
system/backend/app/market_data_repository.py
system/backend/app/market_data_service.py
```

Suggested DuckDB tables:

```text
market_symbols
market_prices_daily
data_ingestion_log
```

Suggested initial tickers:

```text
AAPL
MSFT
SPY
NVDA
```

Suggested endpoints:

```text
GET /market/search?q=AAPL
GET /market/stocks/{ticker}
GET /market/prices/{ticker}
POST /market/ingest/{ticker}
```

Data should be point-in-time friendly and should avoid hidden look-ahead assumptions.""",
            """- Docker Compose starts.
- Backend health responds.
- DuckDB file is created or reused.
- Market data tables are created if missing.
- AAPL can be ingested or loaded.
- AAPL can be read back through API.
- Runtime data remains ignored by Git.
- No RL training starts during app startup.""",
            """- Initial market data module exists.
- DuckDB tables exist or are created on demand.
- AAPL daily prices can be persisted.
- Backend API returns stock metadata.
- Backend API returns daily prices.
- Frontend does not access DuckDB directly.
- Structure is compatible with later FinRL/yfinance workflow.
- No heavy training or notebooks run during app startup.""",
            "This is the real foundation for the next application work. Keep it small, but make it real.",
        ),
    },
    {
        "title": "Create Zero Sum inspired stock lookup and stock detail view",
        "milestone": MILESTONE_M2,
        "labels": ["stock-lookup", "zero-sum-public", "frontend", "backend", "market-data", "application", "poc"],
        "category": CATEGORY_DEVELOPMENT,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-14",
        "progress": 0,
        "body": body(
            "Create a first stock lookup and stock detail page inspired by Zero Sum but implemented in the current PoC stack.",
            """A logged-in user should be able to search for a ticker, open a stock detail view and inspect basic market data.

The design may be inspired by Zero Sum's `/stocks/AAPL`, `/chart` and ticker search flows, but the implementation must use FastAPI/Jinja2/DuckDB and the existing local app structure.""",
            """- Application track: yes, because this is user-facing DSS functionality.
- Research track: indirect, because later signals, recommendations and uncertainty may appear here.
- Shared track: yes, because it relies on backend market data and frontend rendering.""",
            """- Fast layer: yes. This is an online user-facing lookup page.
- Slow layer: no. No model training or backtesting should run.""",
            COMMON_SYSTEM_CONTEXT,
            """Suggested frontend routes:

```text
/stocks
/stocks/{ticker}
```

Suggested UI elements:

```text
- ticker search
- stock title and ticker
- latest price / latest close
- simple metadata
- basic price-history placeholder or table
- actions:
  - add to watchlist
  - add to portfolio
  - compare
```

Suggested backend dependencies:

```text
GET /market/search?q=AAPL
GET /market/stocks/{ticker}
GET /market/prices/{ticker}
```""",
            """- `/stocks` loads for logged-in user.
- Logged-out user is redirected to `/login`.
- Searching for `AAPL` returns a result.
- Opening `/stocks/AAPL` shows stock data.
- Missing ticker gives sensible feedback.
- Backend unavailable gives sensible feedback.
- No direct DuckDB access from frontend.""",
            """- Stock lookup page exists.
- Stock detail page or route exists.
- AAPL works as initial test case.
- Page uses backend API.
- Page does not require React/Next.js.
- Page fits current Jinja2/CSS structure.
- Placeholder chart/table is acceptable for first version.""",
            "This is the first concrete Zero Sum-inspired application page.",
        ),
    },
    {
        "title": "Create minimal portfolio watchlist and transaction flow",
        "milestone": MILESTONE_M2,
        "labels": ["portfolio", "watchlist", "zero-sum-public", "frontend", "backend", "duckdb", "application", "poc"],
        "category": CATEGORY_DEVELOPMENT,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-15",
        "progress": 0,
        "body": body(
            "Create the minimal logged-in portfolio, watchlist and transaction flow for the PoC.",
            """The user needs to be able to build a simple portfolio, save tickers to a watchlist and register buy/sell-like actions.

This is not broker integration and must not imply real trading. It is a local PoC portfolio tracker used for strategy, risk and decision-support demonstrations.

Zero Sum's `/portfolio` and `/watchlist` pages are strong visual/flow references.""",
            """- Application track: yes, because portfolio and watchlist are core user flows.
- Research track: indirect, because portfolio state becomes input to risk/recommendation experiments.
- Shared track: yes, because user data, market data and DuckDB persistence connect.""",
            """- Fast layer: yes. Portfolio/watchlist should be interactive in the app.
- Slow layer: no. No training or backtesting should run.""",
            COMMON_SYSTEM_CONTEXT,
            """Suggested frontend routes:

```text
/portfolio
/watchlist
/transactions
```

Suggested backend endpoints:

```text
GET    /portfolio
POST   /portfolio/holdings
DELETE /portfolio/holdings/{holding_id}

GET    /watchlist
POST   /watchlist/items
DELETE /watchlist/items/{ticker}

GET    /transactions
POST   /transactions
```

Suggested DuckDB tables:

```text
user_watchlist
portfolio_holdings
portfolio_transactions
```

Minimum actions:

```text
- add stock to watchlist
- remove stock from watchlist
- add holding
- register buy-like transaction
- register sell-like transaction or reduce holding
- show simple holdings overview
- show simple total value / P&L placeholder
```""",
            """- Logged-in user can open `/portfolio`.
- Logged-in user can open `/watchlist`.
- Logged-out user redirects to `/login`.
- User can add AAPL to watchlist.
- User can remove AAPL from watchlist.
- User can add AAPL as holding.
- Holding persists after refresh.
- Runtime data remains ignored by Git.""",
            """- Portfolio page exists.
- Watchlist page or section exists.
- Transactions are started or scaffolded.
- Add/remove watchlist works.
- Add holding works.
- Holdings persist locally.
- Basic overview is visible.
- No production broker behavior is implied.""",
            "This is PoC tracking only. UI text should avoid phrases like executing real trades.",
        ),
    },
    {
        "title": "Create comparison and correlation prototype from stored market data",
        "milestone": MILESTONE_M3,
        "labels": ["analytics", "zero-sum-public", "frontend", "backend", "market-data", "research", "application", "poc"],
        "category": CATEGORY_EVALUATION,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-16",
        "progress": 0,
        "body": body(
            "Create a minimal stock comparison and correlation prototype using stored market data.",
            """The DSS should help users compare stocks and reason about diversification/risk.

This task implements a first lightweight version inspired by Zero Sum's `/compare?tickers=AAPL,MSFT` and `/correlation` pages.

The first version can use tables and simple normalized return outputs. Full advanced charts can come later.""",
            """- Application track: yes, because users need comparison/risk views.
- Research track: yes, because correlations and return behavior support thesis evaluation and risk discussion.
- Shared track: yes, because it depends on market data, backend API and frontend rendering.""",
            """- Fast layer: yes. It should run quickly from cached/stored market data.
- Slow layer: indirect. It should not trigger heavy training or backtesting.""",
            COMMON_SYSTEM_CONTEXT,
            """Suggested frontend routes:

```text
/compare
/correlation
```

Suggested backend endpoints:

```text
GET /analytics/compare?tickers=AAPL,MSFT&period=1y
GET /analytics/correlation?tickers=AAPL,MSFT,SPY,NVDA&period=1y
```

Suggested calculations:

```text
- adjusted close series
- simple returns
- normalized price series
- return correlation matrix
- basic return/volatility table
```""",
            """- `/compare` loads.
- `/correlation` loads.
- AAPL/MSFT comparison works.
- AAPL/MSFT/SPY/NVDA correlation works.
- Missing ticker gives sensible feedback.
- Insufficient data gives sensible feedback.
- Calculations are fast.
- No frontend DuckDB access.""",
            """- User can compare at least two tickers.
- User can compute correlation for multiple tickers.
- Backend returns comparison data.
- Backend returns correlation matrix.
- Frontend displays results.
- No heavy training/backtesting is triggered.
- Output can support later report figures.""",
            "This is an early decision-support feature, not a complete quant platform.",
        ),
    },
    {
        "title": "Create market overview placeholders for heatmap scanner and technical views",
        "milestone": MILESTONE_M3,
        "labels": ["zero-sum-public", "frontend-reference", "analytics", "frontend", "application", "poc"],
        "category": CATEGORY_DEVELOPMENT,
        "priority": PRIORITY_LOW,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_FUTURE,
        "deadline": "2026-05-18",
        "progress": 0,
        "body": body(
            "Create lightweight placeholders and implementation notes for later market overview features.",
            """Zero Sum has strong market overview pages: `/heatmap`, `/bubble`, `/scanner`, `/technical`, `/sectors`, `/earnings` and `/news`.

These are useful for the long-term DSS, but they should not block the immediate stock/portfolio/market-data path.

This issue creates placeholders and notes so the ideas are captured without expanding the V1.0 scope too much.""",
            """- Application track: yes, because these are future user-facing pages.
- Research track: indirect, because some indicators may later support strategy and risk features.
- Shared track: yes, because backend/data needs should be documented.""",
            """- Fast layer: yes, as future cached online views.
- Slow layer: indirect, because some indicators may be produced by scheduled jobs.""",
            COMMON_SYSTEM_CONTEXT,
            """Create placeholder frontend routes or dashboard cards for:

```text
/heatmap
/bubble
/scanner
/technical
/sectors
/news
/earnings
```

For each, document:

```text
- user value
- required data
- possible backend endpoint
- whether it is V1.0, V1.1 or future work
```

Do not implement full advanced charting in this task.""",
            """- Placeholder links/cards exist or are documented.
- Each feature has a short implementation note.
- Dashboard is not broken.
- No heavy API scraping is introduced.
- No scope creep into full charting platform.""",
            """- Future market overview features are captured.
- V1.0 vs later scope is clear.
- Dashboard can link to placeholders or docs.
- No major framework change is introduced.
- No advanced charting dependency is added without separate decision.""",
            "This issue keeps good Zero Sum ideas visible while protecting the deadline.",
        ),
    },
]

project_id = None
fields: List[Dict[str, Any]] = []

try:
    project_id = get_project_id()
    fields = get_project_fields()
except Exception as exc:
    print(f"Warning: could not read project metadata: {exc}")

created_or_found = []

for task in TASKS:
    issue = create_or_get_issue(task)
    created_or_found.append(issue)
    add_issue_to_project(issue["url"])
    time.sleep(0.5)

    if project_id:
        item_id = find_project_item_id(task["title"])
        if item_id:
            print("  Updating project fields...")
            update_single_select(project_id, item_id, fields, "Status", STATUS_TODO)
            update_single_select(project_id, item_id, fields, "Category", task["category"])
            update_single_select(project_id, item_id, fields, "Priority", task["priority"])
            update_single_select(project_id, item_id, fields, "Roadmap", task["roadmap"])
            update_single_select(project_id, item_id, fields, "Track", task["track"])
            update_single_select(project_id, item_id, fields, "Percentage", PERCENT_0)
            update_date(project_id, item_id, fields, "Deadline", task["deadline"])
            update_number(project_id, item_id, fields, "Progress Number", task["progress"])
        else:
            print("  Warning: could not find project item after adding issue.")
    else:
        print("  Project metadata unavailable; skipped project field updates.")

    print("")

print("Done. Issues created/found:")
for issue in created_or_found:
    number = issue.get("number")
    suffix = f"#{number}" if number else issue.get("url", "")
    print(f"- {suffix} {issue['title']}")
PY

echo ""
echo "Script 17 done."
echo ""
echo "Recommended next issue:"
echo "  Map zero-sum-public features to StockInvestmentDSS PoC"
