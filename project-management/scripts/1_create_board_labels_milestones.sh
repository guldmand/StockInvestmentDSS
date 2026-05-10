#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 1
# Create GitHub Project board, labels, milestones and fields
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"

PROJECT_TITLE="StockInvestmentDSS PoC Sprint"
PROJECT_DESCRIPTION="GitHub Project board for the StockInvestmentDSS thesis PoC sprint."
PROJECT_README="

# StockInvestmentDSS PoC Sprint

This project board tracks the five-day PoC sprint for the StockInvestmentDSS master's thesis.

## Main goal

Build a working proof-of-concept decision support system for stock investors.

## Core tracks

- PoC implementation
- Data pipeline
- Strategy builder
- Portfolio builder
- Decision engine
- Risk and uncertainty
- Audit / transparency
- Report writing

## Sprint logic

- ✅ Now = must be worked on immediately
- 🔜 Next = upcoming tasks
- 🗓️ Later = bonus / future work

## Stop rule

If V1.0 does not work, no V1.1 work should begin.
"

echo "Checking GitHub auth..."
gh auth status || {
  echo "Login first:"
  echo "  gh auth login --web"
  exit 1
}

echo "Refreshing GitHub auth with project scope..."
gh auth refresh -s project

echo "Creating/updating labels..."

create_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  gh label create "$name" \
    --repo "$GH_REPO" \
    --color "$color" \
    --description "$description" \
    --force >/dev/null

  echo "Label ready: $name"
}

# ============================================================
# Labels
# ============================================================

# Type labels
create_label "type:data" "C5DEF5" "Data pipeline, DuckDB, yfinance, features"
create_label "type:frontend" "BFD4F2" "Frontend, UI, charts, dashboard"
create_label "type:backend" "D4C5F9" "Backend, API, app logic"
create_label "type:rl" "FBCA04" "Reinforcement learning, FinRL, IQN"
create_label "type:report" "0E8A16" "Thesis report writing"
create_label "type:infra" "5319E7" "Infrastructure, k3s, Turing Pi, deployment"
create_label "type:audit" "006B75" "Audit log, point-in-time, transparency"
create_label "type:strategy" "F9D0C4" "Strategy builder, strategy configs"
create_label "type:decision" "D93F0B" "Decision engine and recommendations"
create_label "type:portfolio" "1D76DB" "Portfolio creation and state"
create_label "type:documentation" "0075CA" "README, docs, runbooks"
create_label "type:risk" "B60205" "Risk, uncertainty, CVaR, quantiles"
create_label "type:evaluation" "5319E7" "Backtesting, metrics, results and evaluation"

# Priority labels
create_label "priority:urgent" "B60205" "Must be done immediately"
create_label "priority:high" "D93F0B" "Important for PoC"
create_label "priority:medium" "FBCA04" "Useful but not blocking"
create_label "priority:low" "C2E0C6" "Nice to have"

# Scope labels
create_label "scope:poc" "0E8A16" "Required for PoC"
create_label "scope:bonus" "7057FF" "Bonus if time allows"
create_label "scope:future-work" "C5DEF5" "Future work / perspectives"

# Status labels
create_label "status:blocked" "000000" "Blocked"
create_label "status:needs-decision" "F9D0C4" "Needs a decision before implementation"

echo "Creating milestones..."

create_milestone() {
  local title="$1"
  local due_on="$2"
  local description="$3"

  if gh api "repos/$GH_REPO/milestones" --jq '.[].title' | grep -Fxq "$title"; then
    echo "Milestone already exists: $title"
  else
    gh api "repos/$GH_REPO/milestones" \
      -f title="$title" \
      -f due_on="$due_on" \
      -f description="$description" >/dev/null

    echo "Created milestone: $title"
  fi
}

# ============================================================
# Milestones
# ============================================================

create_milestone "M1 — PoC Foundation" "2026-05-10T23:59:00Z" "Repo, DuckDB, yfinance, minimal app shell and k3s platform check."
create_milestone "M2 — Strategy + Portfolio" "2026-05-11T23:59:00Z" "Strategy builder, predefined strategies, custom strategy and portfolio builder."
create_milestone "M3 — Decision Support" "2026-05-12T23:59:00Z" "Decision engine, transaction costs, triggers and chart view."
create_milestone "M4 — Risk, Uncertainty, Audit" "2026-05-13T23:59:00Z" "Quantiles, uncertainty, CVaR proxy and decision audit log."
create_milestone "M5 — Demo + Report Figures" "2026-05-14T23:59:00Z" "End-to-end demo, screenshots, figures and README4."
create_milestone "M6 — Buffer / v1.1" "2026-05-17T23:59:00Z" "Buffer, fixes, optional deployment and PoC improvements."

echo "Creating or finding GitHub Project..."

PROJECT_NUMBER="$(gh project list --owner "$OWNER" --format json --limit 100 \
  --jq ".projects[] | select(.title == \"$PROJECT_TITLE\") | .number" | head -n 1 || true)"

if [[ -z "${PROJECT_NUMBER:-}" ]]; then
  echo "Project does not exist. Creating: $PROJECT_TITLE"

  PROJECT_NUMBER="$(gh project create \
    --owner "$OWNER" \
    --title "$PROJECT_TITLE" \
    --format json \
    --jq '.number')"

  echo "Created project #$PROJECT_NUMBER"
else
  echo "Project already exists: $PROJECT_TITLE (#$PROJECT_NUMBER)"
fi

echo "Project number: $PROJECT_NUMBER"

echo "Updating project description/readme..."

gh project edit "$PROJECT_NUMBER" \
  --owner "$OWNER" \
  --description "$PROJECT_DESCRIPTION" \
  --readme "$PROJECT_README" >/dev/null || {
    echo "Warning: Could not update project description/readme. Continuing..."
  }

# ============================================================
# Project fields
# ============================================================

echo "Creating project fields..."

field_exists() {
  local field_name="$1"

  gh project field-list "$PROJECT_NUMBER" \
    --owner "$OWNER" \
    --format json \
    --jq ".fields[] | select(.name == \"$field_name\") | .name" | grep -Fxq "$field_name"
}

create_single_select_field() {
  local field_name="$1"
  local options="$2"

  if field_exists "$field_name"; then
    echo "Field already exists: $field_name"
  else
    gh project field-create "$PROJECT_NUMBER" \
      --owner "$OWNER" \
      --name "$field_name" \
      --data-type "SINGLE_SELECT" \
      --single-select-options "$options" >/dev/null

    echo "Created field: $field_name"
  fi
}

create_number_field() {
  local field_name="$1"

  if field_exists "$field_name"; then
    echo "Field already exists: $field_name"
  else
    gh project field-create "$PROJECT_NUMBER" \
      --owner "$OWNER" \
      --name "$field_name" \
      --data-type "NUMBER" >/dev/null

    echo "Created field: $field_name"
  fi
}

create_date_field() {
  local field_name="$1"

  if field_exists "$field_name"; then
    echo "Field already exists: $field_name"
  else
    gh project field-create "$PROJECT_NUMBER" \
      --owner "$OWNER" \
      --name "$field_name" \
      --data-type "DATE" >/dev/null

    echo "Created field: $field_name"
  fi
}

# Field names inspired by your previous GitHub Project board
create_single_select_field "Status" "Backlog,In Progress,Code Review,Done"
create_single_select_field "Category" "🎨 Design,⚙️ Development,📊 Data,📄 Content,📚 Research,🏗️ Architecture,🔐 Security,🤖 RL / AI,🧪 Evaluation,📝 Report"
create_single_select_field "Priority" "🗼 Urgent,⛰️ High,🫣 Medium,🌈 Low"
create_single_select_field "Roadmap" "✅ Now,🔜 Next,🗓️ Later"
create_single_select_field "Track" "PoC,Report,Infra,Research,Bonus,Future Work"

# Visual progress-bar style, similar to your old board
create_single_select_field "Percentage" "□□□□□□□□□□ 0%,■□□□□□□□□□ 10%,■■□□□□□□□□ 20%,■■■□□□□□□□ 30%,■■■■□□□□□□ 40%,■■■■■□□□□□ 50%,■■■■■■□□□□ 60%,■■■■■■■□□□ 70%,■■■■■■■■□□ 80%,■■■■■■■■■□ 90%,■■■■■■■■■■ 100%"

create_date_field "Deadline"

# Optional numeric field if you later want real calculations
create_number_field "Progress Number"

# ============================================================
# Save project info for script 2 and script 3
# ============================================================

mkdir -p .github/scripts-output

cat > .github/scripts-output/project.env <<EOF
OWNER="$OWNER"
REPO="$REPO"
GH_REPO="$GH_REPO"
PROJECT_TITLE="$PROJECT_TITLE"
PROJECT_NUMBER="$PROJECT_NUMBER"
EOF

echo "Saved project info to .github/scripts-output/project.env"

echo ""
echo "Done."
echo ""
echo "Project created/updated:"
echo "  $PROJECT_TITLE (#$PROJECT_NUMBER)"
echo ""
echo "Next step:"
echo "  ./scripts/2_create_issues.sh"