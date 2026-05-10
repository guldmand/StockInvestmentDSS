#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 3 LIGHT
# Optimized project field updater
#
# Sets only:
# - Roadmap
# - Percentage
# - Deadline
#
# Leaves the rest for manual cleanup:
# - Status
# - Category
# - Priority
# - Track
# - Progress Number
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"
PROJECT_NUMBER="11"

# ------------------------------------------------------------
# Read generated project.env safely
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
  PROJECT_TITLE_FROM_FILE="$(read_env_value "PROJECT_TITLE")"
  PROJECT_NUMBER_FROM_FILE="$(read_env_value "PROJECT_NUMBER")"

  [[ -n "${OWNER_FROM_FILE:-}" ]] && OWNER="$OWNER_FROM_FILE"
  [[ -n "${REPO_FROM_FILE:-}" ]] && REPO="$REPO_FROM_FILE"
  [[ -n "${GH_REPO_FROM_FILE:-}" ]] && GH_REPO="$GH_REPO_FROM_FILE"
  [[ -n "${PROJECT_TITLE_FROM_FILE:-}" ]] && PROJECT_TITLE="$PROJECT_TITLE_FROM_FILE"
  [[ -n "${PROJECT_NUMBER_FROM_FILE:-}" ]] && PROJECT_NUMBER="$PROJECT_NUMBER_FROM_FILE"
fi

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE"
echo "Using project number: $PROJECT_NUMBER"

echo "Checking GitHub auth..."
gh auth status || {
  echo "Login first:"
  echo "  gh auth login --web"
  exit 1
}

echo "Refreshing GitHub auth with project scope..."
gh auth refresh -s project

# ------------------------------------------------------------
# Python helper detection
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
  echo "Install Python or make sure python is available in PATH."
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

# ------------------------------------------------------------
# Cache project data to avoid repeated lookup API calls
# ------------------------------------------------------------

CACHE_DIR=".github/scripts-output/project-cache"
mkdir -p "$CACHE_DIR"

FIELDS_JSON="$CACHE_DIR/fields.json"
ITEMS_JSON="$CACHE_DIR/items.json"

echo "Fetching project ID..."

PROJECT_ID="$(gh project view "$PROJECT_NUMBER" \
  --owner "$OWNER" \
  --format json \
  --jq '.id')"

if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: Could not find PROJECT_ID"
  exit 1
fi

echo "Project ID: $PROJECT_ID"

echo "Caching project fields..."
gh project field-list "$PROJECT_NUMBER" \
  --owner "$OWNER" \
  --limit 100 \
  --format json > "$FIELDS_JSON"

echo "Caching project items..."
gh project item-list "$PROJECT_NUMBER" \
  --owner "$OWNER" \
  --limit 100 \
  --format json > "$ITEMS_JSON"

# ------------------------------------------------------------
# Local JSON lookup helpers
# ------------------------------------------------------------

get_field_id() {
  local field_name="$1"

  $PYTHON_BIN - "$FIELDS_JSON" "$field_name" <<'PY'
import json
import sys

path = sys.argv[1]
field_name = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for field in data.get("fields", []):
    if field.get("name") == field_name:
        print(field.get("id", ""))
        sys.exit(0)
PY
}

get_option_id() {
  local field_name="$1"
  local option_name="$2"

  $PYTHON_BIN - "$FIELDS_JSON" "$field_name" "$option_name" <<'PY'
import json
import sys

path = sys.argv[1]
field_name = sys.argv[2]
option_name = sys.argv[3]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for field in data.get("fields", []):
    if field.get("name") == field_name:
        for option in field.get("options", []):
            if option.get("name") == option_name:
                print(option.get("id", ""))
                sys.exit(0)
PY
}

get_item_id_by_title() {
  local title="$1"

  $PYTHON_BIN - "$ITEMS_JSON" "$title" <<'PY'
import json
import sys

path = sys.argv[1]
title = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data.get("items", []):
    content = item.get("content") or {}
    if content.get("title") == title:
        print(item.get("id", ""))
        sys.exit(0)
PY
}

# ------------------------------------------------------------
# Field IDs and option IDs
# ------------------------------------------------------------

ROADMAP_FIELD_ID="$(get_field_id "Roadmap")"
PERCENTAGE_FIELD_ID="$(get_field_id "Percentage")"
DEADLINE_FIELD_ID="$(get_field_id "Deadline")"

if [[ -z "$ROADMAP_FIELD_ID" ]]; then
  echo "ERROR: Roadmap field not found."
  exit 1
fi

if [[ -z "$PERCENTAGE_FIELD_ID" ]]; then
  echo "ERROR: Percentage field not found."
  exit 1
fi

if [[ -z "$DEADLINE_FIELD_ID" ]]; then
  echo "ERROR: Deadline field not found."
  exit 1
fi

NOW="✅ Now"
NEXT="🔜 Next"
LATER="🗓️ Later"
P0="□□□□□□□□□□ 0%"

ROADMAP_NOW_OPTION_ID="$(get_option_id "Roadmap" "$NOW")"
ROADMAP_NEXT_OPTION_ID="$(get_option_id "Roadmap" "$NEXT")"
ROADMAP_LATER_OPTION_ID="$(get_option_id "Roadmap" "$LATER")"
PERCENTAGE_0_OPTION_ID="$(get_option_id "Percentage" "$P0")"

if [[ -z "$ROADMAP_NOW_OPTION_ID" ]]; then
  echo "ERROR: Roadmap option not found: $NOW"
  exit 1
fi

if [[ -z "$ROADMAP_NEXT_OPTION_ID" ]]; then
  echo "ERROR: Roadmap option not found: $NEXT"
  exit 1
fi

if [[ -z "$ROADMAP_LATER_OPTION_ID" ]]; then
  echo "ERROR: Roadmap option not found: $LATER"
  exit 1
fi

if [[ -z "$PERCENTAGE_0_OPTION_ID" ]]; then
  echo "ERROR: Percentage option not found: $P0"
  echo ""
  echo "Fix:"
  echo "Add this exact option manually to the Percentage field:"
  echo "$P0"
  exit 1
fi

# ------------------------------------------------------------
# Update helpers
# ------------------------------------------------------------

set_roadmap() {
  local item_id="$1"
  local roadmap="$2"
  local option_id=""

  case "$roadmap" in
    "$NOW")
      option_id="$ROADMAP_NOW_OPTION_ID"
      ;;
    "$NEXT")
      option_id="$ROADMAP_NEXT_OPTION_ID"
      ;;
    "$LATER")
      option_id="$ROADMAP_LATER_OPTION_ID"
      ;;
    *)
      echo "  WARNING: Unknown roadmap: $roadmap"
      return 0
      ;;
  esac

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$ROADMAP_FIELD_ID" \
    --single-select-option-id "$option_id" >/dev/null

  echo "  Roadmap = $roadmap"
}

set_percentage_0() {
  local item_id="$1"

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$PERCENTAGE_FIELD_ID" \
    --single-select-option-id "$PERCENTAGE_0_OPTION_ID" >/dev/null

  echo "  Percentage = $P0"
}

set_deadline() {
  local item_id="$1"
  local deadline="$2"

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$DEADLINE_FIELD_ID" \
    --date "$deadline" >/dev/null

  echo "  Deadline = $deadline"
}

update_issue_light() {
  local title="$1"
  local roadmap="$2"
  local deadline="$3"

  echo ""
  echo "Updating: $title"

  local item_id
  item_id="$(get_item_id_by_title "$title")"

  if [[ -z "$item_id" ]]; then
    echo "  WARNING: Project item not found: $title"
    return 0
  fi

  set_roadmap "$item_id" "$roadmap"
  set_percentage_0 "$item_id"
  set_deadline "$item_id" "$deadline"
}

echo ""
echo "Updating project fields:"
echo "- Roadmap"
echo "- Percentage"
echo "- Deadline"
echo ""
echo "Other fields will be handled manually:"
echo "- Status"
echo "- Category"
echo "- Priority"
echo "- Track"
echo "- Progress Number"

# ============================================================
# M1 — PoC Foundation — NOW — 2026-05-10
# ============================================================

update_issue_light "Setup PoC repository structure" "$NOW" "2026-05-10"
update_issue_light "Create DuckDB schema for PoC" "$NOW" "2026-05-10"
update_issue_light "Implement yfinance ingestion" "$NOW" "2026-05-10"
update_issue_light "Create minimal web app shell" "$NOW" "2026-05-10"
update_issue_light "Verify k3s platform on Turing Pi" "$NOW" "2026-05-10"
update_issue_light "Write Introduction" "$NOW" "2026-05-10"

# ============================================================
# M2 — Strategy + Portfolio — NEXT — 2026-05-11
# ============================================================

update_issue_light "Define predefined strategy library" "$NEXT" "2026-05-11"
update_issue_light "Implement strategy builder UI" "$NEXT" "2026-05-11"
update_issue_light "Implement portfolio builder" "$NEXT" "2026-05-11"
update_issue_light "Write Background: RL, FinRL, IQN, uncertainty" "$NEXT" "2026-05-11"

# ============================================================
# M3 — Decision Support — LATER — 2026-05-12
# ============================================================

update_issue_light "Implement decision engine v1" "$LATER" "2026-05-12"
update_issue_light "Add transaction cost penalty" "$LATER" "2026-05-12"
update_issue_light "Implement stop-loss and take-profit triggers" "$LATER" "2026-05-12"
update_issue_light "Evaluate Zero Sum frontend integration" "$LATER" "2026-05-12"
update_issue_light "Write System Design" "$LATER" "2026-05-12"

# ============================================================
# M4 — Risk, Uncertainty, Audit — LATER — 2026-05-13
# ============================================================

update_issue_light "Implement quantile-style risk output" "$LATER" "2026-05-13"
update_issue_light "Implement uncertainty score" "$LATER" "2026-05-13"
update_issue_light "Implement decision audit log" "$LATER" "2026-05-13"
update_issue_light "Write Methodology" "$LATER" "2026-05-13"

# ============================================================
# M5 — Demo + Report Figures — LATER — 2026-05-14
# ============================================================

update_issue_light "Create end-to-end demo flow" "$LATER" "2026-05-14"
update_issue_light "Create thesis screenshots and figures" "$LATER" "2026-05-14"
update_issue_light "Write README4.md" "$LATER" "2026-05-14"
update_issue_light "Write Results / Case Demonstration" "$LATER" "2026-05-14"

# ============================================================
# M6 — Buffer / v1.1 — LATER — 2026-05-17
# ============================================================

update_issue_light "Write Discussion and Future Work" "$LATER" "2026-05-17"

echo ""
echo "Done."
echo ""
echo "Updated fields:"
echo "- Roadmap"
echo "- Percentage"
echo "- Deadline"
echo ""
echo "Refresh the GitHub Project view."
echo ""
echo "Expected Roadmap groups:"
echo "  ✅ Now"
echo "  🔜 Next"
echo "  🗓️ Later"