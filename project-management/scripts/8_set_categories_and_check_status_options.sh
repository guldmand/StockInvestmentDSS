#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 8
# Set Category field for all project tasks
# Check Status options for Todo / In Progress / Code Review / Done
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
  echo "Login first:"
  echo "  gh auth login --web"
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

CATEGORY_OPTIONS = {
    "design": "🎨 Design",
    "development": "⚙️ Development",
    "data": "📊 Data",
    "content": "📄 Content",
    "research": "📚 Research",
    "architecture": "🏗️ Architecture",
    "security": "🔐 Security",
    "rl_ai": "🤖 RL / AI",
    "evaluation": "🧪 Evaluation",
    "report": "📝 Report",
}

REQUIRED_STATUS_OPTIONS = [
    "Todo",
    "In Progress",
    "Code Review",
    "Done",
]

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

    if rate["remaining"] < 500:
        print("")
        print("WARNING: GraphQL remaining is low.")
        print("Wait until resetAt if this script fails.")
        print("The script is idempotent and can safely be rerun.")

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

def fetch_all_issues_rest():
    data = gh_json([
        "gh", "api",
        "--paginate",
        "--slurp",
        f"repos/{GH_REPO}/issues?state=all&per_page=100"
    ]) or []

    issues = {}

    for page in data:
        for item in page:
            if "pull_request" in item:
                continue

            issues[item["title"]] = {
                "number": item["number"],
                "title": item["title"],
                "url": item["html_url"],
                "labels": [label["name"] for label in item.get("labels", [])],
            }

    return issues

def field_id(fields, name):
    for field in fields:
        if field.get("name") == name:
            return field.get("id")
    return None

def field_options(fields, field_name):
    for field in fields:
        if field.get("name") == field_name:
            return [option.get("name") for option in field.get("options", [])]
    return []

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
    query = """
    mutation(
      $projectId: ID!,
      $itemId: ID!,
      $fieldId: ID!,
      $optionId: String!
    ) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
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
        "-f", f"fieldId={field_id_value}",
        "-f", f"optionId={option_id_value}",
    ])

def infer_category(issue_title, labels):
    title = issue_title.lower()
    label_set = set(label.lower() for label in labels)

    # Report first, because many report issues also have documentation-like wording
    if (
        "report" in label_set
        or "latex" in label_set
        or "bibtex" in label_set
        or "thesis" in title
        or "readme" in title
        or "introduction" in title
        or "background" in title
        or "methodology" in title
        or "results" in title
        or "discussion" in title
        or "bibtex" in title
        or "report.tex" in title
        or "evidence package" in title
        or "figures" in title
        or "demo script" in title
    ):
        return "report"

    if (
        "rl" in label_set
        or "finrl" in label_set
        or "gymnasium" in label_set
        or "iqn" in label_set
        or "uncertainty" in label_set
        or "training" in label_set
        or "finrl" in title
        or "gymnasium" in title
        or "iqn" in title
        or "evidential" in title
        or "agent" in title
        or "random policy" in title
        or "model registry" in title
        or "pretrained model" in title
    ):
        return "rl_ai"

    if (
        "evaluation" in label_set
        or "test" in label_set
        or "risk" in label_set
        or "metrics" in title
        or "baseline" in title
        or "backtest" in title
        or "test " in title
        or title.startswith("test")
        or "validate" in title
        or "definition of done" in title
        or "v1.0" in title
        or "freeze" in title
    ):
        return "evaluation"

    if (
        "data" in label_set
        or "duckdb" in label_set
        or "api" in label_set
        or "features" in label_set
        or "macro" in label_set
        or "fundamentals" in label_set
        or "guldnas" in label_set
        or "yfinance" in title
        or "duckdb" in title
        or "market data" in title
        or "ingestion" in title
        or "parquet" in title
        or "storage" in title
        or "dataset" in title
        or "snapshot" in title
        or "api response" in title
        or "technical indicator" in title
        or "macro" in title
        or "fundamentals" in title
        or "guldnas" in title
    ):
        return "data"

    if (
        "infra" in label_set
        or "architecture" in label_set
        or "docker" in label_set
        or "k3s" in label_set
        or "devops" in label_set
        or "ci" in label_set
        or "monitoring" in label_set
        or "rollback" in label_set
        or "environment" in label_set
        or "local" in label_set
        or "test-env" in label_set
        or "production" in label_set
        or "worker" in label_set
        or "docker" in title
        or "container" in title
        or "compose" in title
        or "k3s" in title
        or "deployment" in title
        or "rollback" in title
        or "github actions" in title
        or "dependabot" in title
        or "ci workflow" in title
        or "environment configuration" in title
        or ".env" in title
        or "production route" in title
        or "test deployment" in title
        or "slow-layer" in title
        or "fast-layer" in title
        or "architecture" in title
    ):
        return "architecture"

    if (
        "security" in label_set
        or "security" in title
        or "secrets" in title
        or "password" in title
        or "auth" in title
        or "login" in title
    ):
        return "security"

    if (
        "research" in label_set
        or "future-work" in label_set
        or "assumptions" in title
        or "limitations" in title
        or "future" in title
    ):
        return "research"

    if (
        "frontend" in label_set
        or "backend" in label_set
        or "strategy" in label_set
        or "decision" in label_set
        or "portfolio" in label_set
        or "audit" in label_set
        or "frontend" in title
        or "backend" in title
        or "web app" in title
        or "strategy builder" in title
        or "portfolio builder" in title
        or "decision engine" in title
        or "decision card" in title
        or "audit log" in title
        or "custom strategy" in title
        or "locked assets" in title
        or "stock lookup" in title
        or "user creation" in title
    ):
        return "development"

    if (
        "documentation" in label_set
        or "document" in title
        or "docs" in title
    ):
        return "content"

    return "development"

print("")
print("Checking GraphQL rate before running...")
check_graphql_rate_limit()

print("")
print("Fetching project metadata...")
project_id = get_project_id()
fields = get_project_fields()

category_field_id = field_id(fields, "Category")
status_options = field_options(fields, "Status")
category_options = field_options(fields, "Category")

if not category_field_id:
    print("ERROR: Project field not found: Category")
    sys.exit(1)

print("")
print("Checking Status options...")
print("Existing Status options:")
for option in status_options:
    print(f"  - {option}")

missing_status = [option for option in REQUIRED_STATUS_OPTIONS if option not in status_options]

if missing_status:
    print("")
    print("WARNING: Missing Status options:")
    for option in missing_status:
        print(f"  - {option}")
    print("")
    print("GitHub CLI is not reliable for adding options to an existing ProjectV2 single-select Status field.")
    print("Add these manually in the UI:")
    print("  Status field menu -> Edit field -> Add options:")
    for option in missing_status:
        print(f"    {option}")
else:
    print("Status field already has all required options.")

print("")
print("Checking Category options...")
required_category_names = list(CATEGORY_OPTIONS.values())
missing_categories = [name for name in required_category_names if name not in category_options]

if missing_categories:
    print("")
    print("ERROR: Missing Category options:")
    for name in missing_categories:
        print(f"  - {name}")
    print("")
    print("Add them manually to the Category field, or rerun Script 1 if needed.")
    sys.exit(1)

category_option_ids = {
    key: option_id(fields, "Category", value)
    for key, value in CATEGORY_OPTIONS.items()
}

print("")
print("Fetching project items...")
items = get_project_items()

print("")
print("Fetching repository issues...")
issues_by_title = fetch_all_issues_rest()

print("")
print("Setting Category for all project items...")

updated = 0
skipped = 0

for item in items:
    content = item.get("content") or {}
    title = content.get("title")

    if not title:
        skipped += 1
        continue

    issue = issues_by_title.get(title)

    if not issue:
        print(f"Skipping non-issue item: {title}")
        skipped += 1
        continue

    category_key = infer_category(title, issue["labels"])
    category_name = CATEGORY_OPTIONS[category_key]
    option = category_option_ids[category_key]
    item_id = item.get("id")

    if not item_id:
        skipped += 1
        continue

    print(f"Updating Category: #{issue['number']} {title}")
    print(f"  Category -> {category_name}")

    update_single_select(
        project_id=project_id,
        item_id=item_id,
        field_id_value=category_field_id,
        option_id_value=option,
    )

    updated += 1
    time.sleep(0.12)

print("")
print(f"Category updates completed: {updated}")
print(f"Skipped items: {skipped}")

print("")
print("Final GraphQL rate check...")
check_graphql_rate_limit()

print("")
print("Done.")
print("")
print("Script 8 completed.")
print("")
print("What was done:")
print("- Category field set for all GitHub Project issue items")
print("- Status options checked")
print("- Missing Status options reported for manual fix")
PY