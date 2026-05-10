#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 2
# Create issues and add them to the GitHub Project board
# ============================================================

# Default values
OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"
PROJECT_NUMBER=""

# Safely read project.env without sourcing it directly.
# This avoids errors when values contain spaces.
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
[[ -n "${PROJECT_NUMBER:-}" ]] && echo "Using project number: $PROJECT_NUMBER"

echo "Checking GitHub auth..."
gh auth status || {
  echo "Login first:"
  echo "  gh auth login --web"
  exit 1
}

echo "Refreshing GitHub auth with project scope..."
gh auth refresh -s project

issue_exists() {
  local title="$1"

  gh issue list \
    --repo "$GH_REPO" \
    --state all \
    --search "$title in:title" \
    --json title \
    --jq ".[].title" | grep -Fxq "$title"
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
}

echo "Creating PoC issues..."

create_issue \
"Setup PoC repository structure" \
"M1 — PoC Foundation" \
"type:backend,scope:poc,priority:urgent" \
"## Goal
Create the minimal PoC project structure.

## Acceptance criteria
- system/ or app/ PoC folder exists
- requirements.txt exists
- README4.md exists
- .env.example exists
- runtime-data/ is ignored in Git
- project can be started locally

## Notes
This is the foundation for the five-day PoC sprint."

create_issue \
"Create DuckDB schema for PoC" \
"M1 — PoC Foundation" \
"type:data,scope:poc,priority:urgent" \
"## Goal
Create the initial DuckDB schema for the PoC.

## Required tables
- prices
- strategies
- portfolios
- portfolio_positions
- decision_log
- model_outputs

## Acceptance criteria
- init_db.py creates all tables
- database file can be opened locally
- runtime data is not committed to Git
- schema supports point-in-time decision logging"

create_issue \
"Implement yfinance ingestion" \
"M1 — PoC Foundation" \
"type:data,scope:poc,priority:urgent" \
"## Goal
Fetch stock data from yfinance and store it in DuckDB.

## Acceptance criteria
- fetch AAPL, MSFT and NVDA
- store OHLCV in DuckDB
- reload without duplicate rows
- data can be queried by ticker and date

## Output
ticker → yfinance → DuckDB works."

create_issue \
"Create minimal web app shell" \
"M1 — PoC Foundation" \
"type:frontend,scope:poc,priority:urgent" \
"## Goal
Create the minimal web interface shell.

## Acceptance criteria
- password gate exists
- dashboard loads
- app can call backend or load DuckDB-backed data
- navigation exists for strategy, portfolio and decisions

## Notes
Keep frontend minimal. The PoC must work before polishing UI."

create_issue \
"Verify k3s platform on Turing Pi" \
"M1 — PoC Foundation" \
"type:infra,scope:poc,priority:urgent" \
"## Goal
Verify the Turing Pi k3s cluster as the intended PoC runtime platform.

## Acceptance criteria
- kubectl get nodes -o wide works
- namespace stockinvestmentdss created
- docs/infrastructure/k3s-poc-status.md written
- NAS/DuckDB storage path described
- minimal deployment target documented

## Commands
- kubectl get nodes -o wide
- kubectl create namespace stockinvestmentdss

## Scope
This is a minimal platform check, not full production Kubernetes."

create_issue \
"Define predefined strategy library" \
"M2 — Strategy + Portfolio" \
"type:strategy,scope:poc,priority:urgent" \
"## Goal
Define the predefined long-only strategy candidates.

## Strategies
1. Conservative Long-Only
2. Balanced Long-Only
3. Aggressive Growth
4. Momentum
5. Value / Quality
6. Profit Protection
7. Low Turnover
8. Tech Growth
9. Agent-Derived Candidate

## Acceptance criteria
- 6–9 strategies defined as JSON/YAML
- all strategies are long-only
- each strategy has risk score
- each strategy has max position size
- each strategy has stop-loss/take-profit settings
- each strategy has rebalance frequency"

create_issue \
"Implement strategy builder UI" \
"M2 — Strategy + Portfolio" \
"type:frontend,type:strategy,scope:poc,priority:urgent" \
"## Goal
Build the guided strategy builder.

## Acceptance criteria
- risk slider exists
- strategy dropdown exists
- custom strategy panel appears when custom is selected
- strategy JSON is saved to DuckDB
- strategy can be loaded again

## Custom strategy fields
- risk score
- investment horizon
- max position size
- cash buffer
- stop-loss
- take-profit
- rebalance frequency
- preferred style
- locked assets"

create_issue \
"Implement portfolio builder" \
"M2 — Strategy + Portfolio" \
"type:portfolio,scope:poc,priority:urgent" \
"## Goal
Allow the user to create a portfolio.

## Acceptance criteria
- investor name
- initial capital
- ticker positions
- shares
- buy price
- buy date
- current price
- current value
- gain/loss
- portfolio state can be passed to decision engine"

create_issue \
"Implement decision engine v1" \
"M3 — Decision Support" \
"type:decision,scope:poc,priority:urgent" \
"## Goal
Generate decision alternatives for the current portfolio and strategy.

## Required alternatives
- HOLD
- REDUCE
- SELL
- REBALANCE
- SWITCH STRATEGY

## Acceptance criteria
- decision cards are generated
- each card has explanation
- each card references strategy constraints
- decision engine works without full RL training"

create_issue \
"Add transaction cost penalty" \
"M3 — Decision Support" \
"type:decision,type:rl,scope:poc,priority:high" \
"## Goal
Model trading costs so the system discourages overtrading.

## Acceptance criteria
- fixed transaction cost parameter exists
- turnover penalty is included
- decision cards explain the cost effect
- methodology can be described in report

## Formula idea
reward = portfolio_change - transaction_cost - risk_penalty"

create_issue \
"Implement stop-loss and take-profit triggers" \
"M3 — Decision Support" \
"type:strategy,type:decision,scope:poc,priority:high" \
"## Goal
Implement basic algorithmic trading trigger rules.

## Acceptance criteria
- stop-loss trigger fires based on current PnL
- take-profit trigger fires based on current PnL
- trigger creates decision alternative
- user can accept or ignore the trigger
- trigger is logged in decision audit trail"

create_issue \
"Evaluate Zero Sum frontend integration" \
"M3 — Decision Support" \
"type:frontend,scope:poc,priority:medium" \
"## Goal
Evaluate whether tristcoil/zero-sum-public can be reused or used as frontend inspiration.

## Acceptance criteria
- repo cloned or inspected
- integration path chosen:
  A. component reuse
  B. iframe/link/screenshot
  C. own simple chart
- decision documented in README4 or docs/frontend.md

## Stop rule
This must not block the PoC. If integration is slow, use a simple chart fallback."

create_issue \
"Implement quantile-style risk output" \
"M4 — Risk, Uncertainty, Audit" \
"type:rl,type:risk,scope:poc,priority:urgent" \
"## Goal
Expose IQN-inspired quantile outputs.

## Acceptance criteria
- q10 displayed
- q50 displayed
- q90 displayed
- downside risk shown
- CVaR proxy shown
- output appears on decision cards

## Notes
Can be implemented as IQN-style/proxy output for V1.0 and replaced by stronger IQN later."

create_issue \
"Implement uncertainty score" \
"M4 — Risk, Uncertainty, Audit" \
"type:risk,type:rl,scope:poc,priority:high" \
"## Goal
Expose uncertainty in decision support output.

## Acceptance criteria
- uncertainty label: low / medium / high
- explanation included
- uncertainty shown beside decision alternatives
- implementation can later be replaced by evidential model

## Report angle
This supports uncertainty-aware decision support."

create_issue \
"Implement decision audit log" \
"M4 — Risk, Uncertainty, Audit" \
"type:audit,scope:poc,priority:urgent" \
"## Goal
Implement point-in-time decision logging.

## Acceptance criteria
- input market state logged
- portfolio state logged
- strategy config logged
- suggestions logged
- model/risk output logged
- user choice logged
- timestamp logged
- decision can be reconstructed

## Thesis angle
Bitcoin-inspired transparency, auditability and trust."

create_issue \
"Create end-to-end demo flow" \
"M5 — Demo + Report Figures" \
"scope:poc,priority:urgent" \
"## Goal
Create one complete demo path.

## Acceptance criteria
- login
- create investor
- select strategy
- create portfolio
- fetch data
- show chart
- show decision cards
- switch strategy
- save decision
- show decision log

## Demo case
Use NVDA / AAPL / MSFT style case."

create_issue \
"Create thesis screenshots and figures" \
"M5 — Demo + Report Figures" \
"type:report,scope:poc,priority:urgent" \
"## Goal
Create figures and screenshots for the thesis report.

## Acceptance criteria
- architecture figure
- strategy builder screenshot
- portfolio screenshot
- decision card screenshot
- audit log screenshot
- at least one results table or case table"

create_issue \
"Write README4.md" \
"M5 — Demo + Report Figures" \
"type:documentation,scope:poc,priority:urgent" \
"## Goal
Write the updated README4.md for the PoC.

## Acceptance criteria
- project purpose
- PoC scope
- architecture
- how to run
- demo flow
- k3s note
- Zero Sum frontend note
- future work
- five-day build plan"

create_issue \
"Write Introduction" \
"M1 — PoC Foundation" \
"type:report,priority:urgent" \
"## Goal
Write the thesis introduction draft.

## Acceptance criteria
- motivation
- problem
- research aim
- contribution
- decision support framing
- not autonomous trading bot"

create_issue \
"Write Background: RL, FinRL, IQN, uncertainty" \
"M2 — Strategy + Portfolio" \
"type:report,priority:urgent" \
"## Goal
Write the technical background section.

## Acceptance criteria
- RL explained
- FinRL explained
- distributional RL explained
- IQN explained at intuition level
- evidential uncertainty explained at intuition level
- finance/risk metrics introduced"

create_issue \
"Write System Design" \
"M3 — Decision Support" \
"type:report,priority:urgent" \
"## Goal
Write the system design chapter.

## Acceptance criteria
- strategy abstraction layer
- portfolio state
- decision alternatives
- offline training / online decision support split
- auditability
- point-in-time correctness
- human-in-the-loop framing"

create_issue \
"Write Methodology" \
"M4 — Risk, Uncertainty, Audit" \
"type:report,priority:high" \
"## Goal
Write methodology chapter.

## Acceptance criteria
- data pipeline
- feature generation
- reward / transaction cost
- risk metrics
- quantile output
- uncertainty output
- PoC assumptions and limitations"

create_issue \
"Write Results / Case Demonstration" \
"M5 — Demo + Report Figures" \
"type:report,priority:high" \
"## Goal
Write the results and case demonstration section.

## Acceptance criteria
- one portfolio case
- one strategy switch case
- one sell/reduce/hold case
- screenshots or figures included
- decision support output explained"

create_issue \
"Write Discussion and Future Work" \
"M6 — Buffer / v1.1" \
"type:report,priority:high" \
"## Goal
Write discussion and future work.

## Acceptance criteria
- limitations
- multi-agent strategy discovery
- real user feedback loop
- full k3s/homelab deployment
- compliance and trust
- stronger FinRL/IQN/evidential implementation"

echo ""
echo "Done."
echo "Issues created and added to project: $PROJECT_TITLE"