#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 9
# Add missing baseline comparison issue
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"

PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

TITLE="Create risk-adjusted and equal-weight baseline comparison"
MILESTONE="M4 — Risk, Uncertainty, Audit"

LABELS="evaluation,risk,rl,report"

BODY=$(cat <<'EOF'
## Goal

Create a simple baseline comparison that makes the PoC results interpretable in the thesis.

The purpose is to compare the decision-support/RL-oriented output against understandable portfolio baselines.

## Baselines to compare

- Equal-weight portfolio
- Buy-and-hold portfolio
- Risk-adjusted portfolio / decision-support output

## Acceptance criteria

- Equal-weight baseline is computed
- Buy-and-hold baseline is computed
- Risk-adjusted output is computed or proxied from the PoC decision engine
- Comparison includes at least:
  - cumulative return
  - volatility
  - Sharpe-style metric
  - max drawdown or downside-risk proxy
  - CVaR / Expected Shortfall proxy if available
- Results are stored in DuckDB or exported as CSV/Parquet
- Result table can be used directly in the thesis report
- Limitations are documented clearly

## Thesis relevance

This task supports the Results / Case Demonstration section by giving the PoC a meaningful benchmark.

It also helps avoid presenting the RL/decision-support output in isolation.

## Notes

This does not need to be a perfect production-grade backtest for V1.0.

For the PoC, the key is to show a transparent comparison between:

```text
simple baseline
vs.
risk-aware decision-support logic
vs.
later RL/FinRL model output
```
EOF
)

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE (#$PROJECT_NUMBER)"
echo ""

echo "Checking GitHub auth..."
gh auth status

echo ""
echo "Creating/finding issue..."

EXISTING_NUMBER="$(
  gh issue list \
    --repo "$GH_REPO" \
    --state all \
    --search "\"$TITLE\" in:title" \
    --json number,title \
    --jq ".[] | select(.title == \"$TITLE\") | .number" \
  | head -n 1 || true
)"

if [[ -n "${EXISTING_NUMBER:-}" ]]; then
  ISSUE_NUMBER="$EXISTING_NUMBER"
  echo "Issue already exists: #$ISSUE_NUMBER"
else
  ISSUE_URL="$(
    gh issue create \
      --repo "$GH_REPO" \
      --title "$TITLE" \
      --body "$BODY" \
      --label "$LABELS" \
      --milestone "$MILESTONE"
  )"

  ISSUE_NUMBER="${ISSUE_URL##*/}"
  echo "Created issue: #$ISSUE_NUMBER"
fi

ISSUE_URL="$(gh issue view "$ISSUE_NUMBER" --repo "$GH_REPO" --json url --jq '.url')"

echo ""
echo "Adding issue to project..."

gh project item-add "$PROJECT_NUMBER" \
  --owner "$OWNER" \
  --url "$ISSUE_URL" >/dev/null || true

echo "Issue added/found in project."

echo ""
echo "Setting project fields..."

PYTHON_BIN="python"
if ! command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PYTHONIOENCODING=utf-8 "$PYTHON_BIN" <<'PY'
import json
import subprocess
import sys

OWNER = "guldmand"
PROJECT_NUMBER = "11"
TITLE = "Create risk-adjusted and equal-weight baseline comparison"

FIELD_VALUES = {
    "Status": "Todo",
    "Category": "🧪 Evaluation",
    "Priority": "⛰️ High",
    "Roadmap": "🗓️ Later",
    "Percentage": "□□□□□□□□□□ 0%",
}

DATE_VALUES = {
    "Deadline": "2026-05-13",
}

NUMBER_VALUES = {
    "Progress Number": 0,
}

def run(cmd):
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )

    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit(result.returncode)

    return result.stdout.strip()

def gh_json(cmd):
    out = run(cmd)
    return json.loads(out) if out else {}

project = gh_json([
    "gh", "project", "view", PROJECT_NUMBER,
    "--owner", OWNER,
    "--format", "json"
])

project_id = project["id"]

fields_data = gh_json([
    "gh", "project", "field-list", PROJECT_NUMBER,
    "--owner", OWNER,
    "--format", "json",
    "--limit", "100"
])

fields = {field["name"]: field for field in fields_data["fields"]}

items_data = gh_json([
    "gh", "project", "item-list", PROJECT_NUMBER,
    "--owner", OWNER,
    "--format", "json",
    "--limit", "200"
])

item_id = None

for item in items_data["items"]:
    content = item.get("content") or {}
    if content.get("title") == TITLE:
        item_id = item["id"]
        break

if not item_id:
    print(f"Could not find project item for: {TITLE}")
    sys.exit(1)

def set_single_select(field_name, option_name):
    field = fields.get(field_name)

    if not field:
        print(f"WARNING: Field not found: {field_name}")
        return

    option_id = None

    for option in field.get("options", []):
        if option.get("name") == option_name:
            option_id = option["id"]
            break

    if not option_id:
        print(f"WARNING: Option not found: {field_name} -> {option_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--single-select-option-id", option_id,
    ])

    print(f"Set {field_name} = {option_name}")

def set_date(field_name, value):
    field = fields.get(field_name)

    if not field:
        print(f"WARNING: Field not found: {field_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--date", value,
    ])

    print(f"Set {field_name} = {value}")

def set_number(field_name, value):
    field = fields.get(field_name)

    if not field:
        print(f"WARNING: Field not found: {field_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--number", str(value),
    ])

    print(f"Set {field_name} = {value}")

for field_name, option_name in FIELD_VALUES.items():
    set_single_select(field_name, option_name)

for field_name, value in DATE_VALUES.items():
    set_date(field_name, value)

for field_name, value in NUMBER_VALUES.items():
    set_number(field_name, value)

print("")
print("Done.")
PY

echo ""
echo "Script 9 completed."
echo ""
echo "Created/updated:"
echo "  #$ISSUE_NUMBER — $TITLE"
