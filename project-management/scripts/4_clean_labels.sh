#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 4
# Clean scoped/meta labels
#
# Converts labels such as:
#   type:backend       -> backend
#   scope:poc          -> poc
#   priority:urgent    -> urgent
#
# It does NOT recreate issues.
# It updates existing issues by replacing labels.
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"

echo "Using repo: $GH_REPO"

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

replace_label_on_issues() {
  local old_label="$1"
  local new_label="$2"

  echo ""
  echo "Replacing label:"
  echo "  $old_label -> $new_label"

  local issue_numbers
  issue_numbers="$(gh issue list \
    --repo "$GH_REPO" \
    --state all \
    --label "$old_label" \
    --limit 200 \
    --json number \
    --jq '.[].number' || true)"

  if [[ -z "${issue_numbers:-}" ]]; then
    echo "  No issues found with label: $old_label"
  else
    while IFS= read -r issue_number; do
      [[ -z "$issue_number" ]] && continue

      echo "  Updating issue #$issue_number"

      gh issue edit "$issue_number" \
        --repo "$GH_REPO" \
        --add-label "$new_label" \
        --remove-label "$old_label" >/dev/null
    done <<< "$issue_numbers"
  fi
}

delete_old_label_if_exists() {
  local old_label="$1"

  if gh label list --repo "$GH_REPO" --limit 200 --json name --jq '.[].name' | grep -Fxq "$old_label"; then
    echo "Deleting old label: $old_label"
    gh label delete "$old_label" --repo "$GH_REPO" --yes >/dev/null || {
      echo "  Warning: Could not delete label: $old_label"
    }
  else
    echo "Old label already gone: $old_label"
  fi
}

echo ""
echo "Creating clean labels..."

# Core work labels
create_label_if_missing "data" "C5DEF5" "Data pipeline, DuckDB, yfinance, features"
create_label_if_missing "frontend" "BFD4F2" "Frontend, UI, charts, dashboard"
create_label_if_missing "backend" "D4C5F9" "Backend, API, app logic"
create_label_if_missing "rl" "FBCA04" "Reinforcement learning, FinRL, IQN"
create_label_if_missing "report" "0E8A16" "Thesis report writing"
create_label_if_missing "infra" "5319E7" "Infrastructure, k3s, Turing Pi, deployment"
create_label_if_missing "audit" "006B75" "Audit log, point-in-time, transparency"
create_label_if_missing "strategy" "F9D0C4" "Strategy builder, strategy configs"
create_label_if_missing "decision" "D93F0B" "Decision engine and recommendations"
create_label_if_missing "portfolio" "1D76DB" "Portfolio creation and state"
create_label_if_missing "documentation" "0075CA" "README, docs, runbooks"
create_label_if_missing "risk" "B60205" "Risk, uncertainty, CVaR, quantiles"
create_label_if_missing "evaluation" "5319E7" "Backtesting, metrics, results and evaluation"

# Container/platform labels
create_label_if_missing "docker" "2496ED" "Dockerfiles, Docker Compose and container setup"
create_label_if_missing "k3s" "5319E7" "k3s and Turing Pi deployment target"
create_label_if_missing "worker" "D4C5F9" "Background workers and scheduled jobs"
create_label_if_missing "training" "FBCA04" "Training jobs and model execution"

# Priority labels
create_label_if_missing "urgent" "B60205" "Must be done immediately"
create_label_if_missing "high" "D93F0B" "Important for PoC"
create_label_if_missing "medium" "FBCA04" "Useful but not blocking"
create_label_if_missing "low" "C2E0C6" "Nice to have"

# Scope/status labels
create_label_if_missing "poc" "0E8A16" "Required for PoC"
create_label_if_missing "bonus" "7057FF" "Bonus if time allows"
create_label_if_missing "future-work" "C5DEF5" "Future work / perspectives"
create_label_if_missing "blocked" "000000" "Blocked"
create_label_if_missing "needs-decision" "F9D0C4" "Needs a decision before implementation"

echo ""
echo "Replacing labels on existing issues..."

replace_label_on_issues "type:data" "data"
replace_label_on_issues "type:frontend" "frontend"
replace_label_on_issues "type:backend" "backend"
replace_label_on_issues "type:rl" "rl"
replace_label_on_issues "type:report" "report"
replace_label_on_issues "type:infra" "infra"
replace_label_on_issues "type:audit" "audit"
replace_label_on_issues "type:strategy" "strategy"
replace_label_on_issues "type:decision" "decision"
replace_label_on_issues "type:portfolio" "portfolio"
replace_label_on_issues "type:documentation" "documentation"
replace_label_on_issues "type:risk" "risk"
replace_label_on_issues "type:evaluation" "evaluation"

replace_label_on_issues "priority:urgent" "urgent"
replace_label_on_issues "priority:high" "high"
replace_label_on_issues "priority:medium" "medium"
replace_label_on_issues "priority:low" "low"

replace_label_on_issues "scope:poc" "poc"
replace_label_on_issues "scope:bonus" "bonus"
replace_label_on_issues "scope:future-work" "future-work"

replace_label_on_issues "status:blocked" "blocked"
replace_label_on_issues "status:needs-decision" "needs-decision"

echo ""
echo "Deleting old scoped labels..."

delete_old_label_if_exists "type:data"
delete_old_label_if_exists "type:frontend"
delete_old_label_if_exists "type:backend"
delete_old_label_if_exists "type:rl"
delete_old_label_if_exists "type:report"
delete_old_label_if_exists "type:infra"
delete_old_label_if_exists "type:audit"
delete_old_label_if_exists "type:strategy"
delete_old_label_if_exists "type:decision"
delete_old_label_if_exists "type:portfolio"
delete_old_label_if_exists "type:documentation"
delete_old_label_if_exists "type:risk"
delete_old_label_if_exists "type:evaluation"

delete_old_label_if_exists "priority:urgent"
delete_old_label_if_exists "priority:high"
delete_old_label_if_exists "priority:medium"
delete_old_label_if_exists "priority:low"

delete_old_label_if_exists "scope:poc"
delete_old_label_if_exists "scope:bonus"
delete_old_label_if_exists "scope:future-work"

delete_old_label_if_exists "status:blocked"
delete_old_label_if_exists "status:needs-decision"

echo ""
echo "Done."
echo "Labels have been cleaned."
echo ""
echo "Refresh the GitHub Project board."