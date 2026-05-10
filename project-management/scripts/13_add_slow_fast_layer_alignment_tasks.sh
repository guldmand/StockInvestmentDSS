#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 13
# Add slow/fast layer alignment tasks
# ============================================================
#
# Purpose:
#   Adds the final alignment tasks that make the README/project board
#   consistent with the slow-layer / fast-layer architecture.
#
#   This script only adds/fixes targeted slow/fast tasks.
#   It does not rewrite all existing issues.
#
# Requirements:
#   - GitHub CLI installed
#   - gh authenticated with repo + project scopes
#   - Existing GitHub Project v2 board
#
# Usage from repository root:
#   bash ./project-management/scripts/13_add_slow_fast_layer_alignment_tasks.sh
#
# Windows PowerShell:
#   & "C:\Program Files\Git\bin\bash.exe" ".\project-management\scripts\13_add_slow_fast_layer_alignment_tasks.sh"
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE (#$PROJECT_NUMBER)"
echo ""

echo "Checking GitHub auth..."
gh auth status

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

create_label "slow-layer" "5319E7" "Offline training, backtesting, evaluation and model registry"
create_label "fast-layer" "0E8A16" "Online decision support, inference, constraints and audit"
create_label "research" "BFD4F2" "Research notebooks, experiments and thesis evidence"
create_label "notebook" "C5DEF5" "Jupyter notebook work"
create_label "experiment" "FBCA04" "Reproducible experiment work"
create_label "wandb" "F9D0C4" "Weights and Biases logging"
create_label "model-registry" "D4C5F9" "Model registry, checkpoints and metadata"
create_label "contract" "7057FF" "Interface or data contract"
create_label "integration" "1D76DB" "Integration between system parts"
create_label "architecture" "5319E7" "Architecture and system design"
create_label "documentation" "0075CA" "Documentation and README alignment"
create_label "evaluation" "0E8A16" "Evaluation, validation and reproducibility"
create_label "decision" "D93F0B" "Decision engine and decision-support output"
create_label "audit" "006B75" "Audit log, transparency and reproducibility"
create_label "high" "D93F0B" "High priority"
create_label "medium" "FBCA04" "Medium priority"

echo ""
echo "Using Python for issue creation and project field updates..."

PYTHON_BIN="python"
if ! command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PYTHONIOENCODING=utf-8 "$PYTHON_BIN" <<'PY'
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

OWNER = "guldmand"
REPO = "StockInvestmentDSS"
GH_REPO = f"{OWNER}/{REPO}"
PROJECT_NUMBER = "11"

STATUS_TODO = "Todo"
PERCENT_0 = "□□□□□□□□□□ 0%"

CATEGORY_ARCHITECTURE = "🏗️ Architecture"
CATEGORY_RESEARCH = "📚 Research"
CATEGORY_DEVELOPMENT = "⚙️ Development"
CATEGORY_EVALUATION = "🧪 Evaluation"
CATEGORY_REPORT = "📝 Report"

PRIORITY_URGENT = "🗼 Urgent"
PRIORITY_HIGH = "⛰️ High"
PRIORITY_MEDIUM = "🫣 Medium"
PRIORITY_LOW = "🌈 Low"

ROADMAP_NOW = "✅ Now"
ROADMAP_NEXT = "🔜 Next"
ROADMAP_LATER = "🗓️ Later"

TRACK_RESEARCH = "Research"
TRACK_POC = "PoC"
TRACK_INFRA = "Infra"

TASKS: List[Dict[str, Any]] = [
    {
        "title": "Define slow-fast research folder convention",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["research", "architecture", "documentation", "slow-layer", "fast-layer"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-11",
        "body": """## Goal

Define how the slow-layer / fast-layer architecture should be represented inside `research/notebooks/` and `research/experiments/`.

## Description

The README defines two operational speeds:

```text
Slow layer = offline model training, backtesting and evaluation
Fast layer = near real-time decision support using available data, cached features and existing models
```

This distinction must also be visible in the research workspace, not only in the application code.

## Implementation

Propose and document a convention such as:

```text
research/notebooks/
├─ 01_slow_layer_finrl_baseline.ipynb
├─ 02_slow_layer_gymnasium_env.ipynb
├─ 03_slow_layer_baseline_comparison.ipynb
├─ 04_fast_layer_decision_case.ipynb
├─ 05_fast_layer_strategy_constraints.ipynb
├─ 06_fast_layer_audit_trace.ipynb
└─ 09_thesis_figures.ipynb

research/experiments/
├─ slow_layer/
│  ├─ finrl_baseline/
│  ├─ buy_and_hold/
│  ├─ equal_weight/
│  ├─ risk_adjusted_baseline/
│  └─ model_registry_export/
└─ fast_layer/
   ├─ decision_case/
   ├─ strategy_constraints/
   ├─ risk_output/
   └─ audit_reproducibility/
```

## Acceptance Criteria

- Research folder convention is documented
- Slow-layer research artifacts are clearly separated from fast-layer artifacts
- Naming convention is compatible with README.md
- Naming convention is simple enough for GitHub Copilot / AI agents to follow
- No heavy implementation is required in this task

## Notes

This is an alignment task. Do not over-engineer it.
""",
    },
    {
        "title": "Create slow-layer research notebook skeleton",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["research", "notebook", "slow-layer", "finrl", "evaluation"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-11",
        "body": """## Goal

Create the initial slow-layer notebook skeleton used for offline training, baseline comparison and thesis evidence.

## Description

This notebook should not try to solve all RL training. It should establish the reproducible research flow for the slow layer.

## Implementation

Create or prepare a notebook such as:

```text
research/notebooks/01_slow_layer_finrl_baseline.ipynb
```

Recommended sections:

```text
1. Purpose
2. Imports and configuration
3. Data loading
4. Point-in-time data assumptions
5. Baseline setup
6. FinRL placeholder / baseline agent setup
7. Metrics
8. W&B logging placeholder
9. Export to research/results/
10. Notes for thesis report
```

## Acceptance Criteria

- Notebook file exists
- Notebook clearly states its slow-layer purpose
- It has a reproducible structure
- It references where outputs should be stored
- It does not require full training to be completed in V1.0
- W&B logging is noted as part of the training/evaluation flow

## Notes

For V1.0 this may be a skeleton plus a minimal baseline run.
""",
    },
    {
        "title": "Create slow-layer reproducible experiment structure",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "experiment", "slow-layer", "evaluation", "devops"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Create a reproducible slow-layer experiment structure under `research/experiments/`.

## Description

The thesis needs experiments that can be rerun or at least explained clearly. This task creates the structure for repeatable offline experiments.

## Implementation

Create a structure such as:

```text
research/experiments/slow_layer/
├─ finrl_baseline/
│  ├─ config.yaml
│  ├─ run.py
│  └─ README.md
├─ buy_and_hold/
├─ equal_weight/
├─ risk_adjusted_baseline/
└─ model_registry_export/
```

Each experiment folder should explain:

- input data
- config file
- output location
- expected metrics
- W&B run naming if applicable

## Acceptance Criteria

- `research/experiments/slow_layer/` exists
- At least one baseline experiment folder exists
- Each created folder has a short README or placeholder
- Output path is documented
- Structure supports W&B logging later
- Structure supports export of metrics/checkpoints

## Notes

Keep this lightweight. The structure matters more than a perfect experiment runner for V1.0.
""",
    },
    {
        "title": "Create fast-layer decision case notebook skeleton",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "notebook", "fast-layer", "decision", "strategy"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Create a notebook that demonstrates the fast-layer decision-support case.

## Description

The fast layer must show how the system can generate a decision now, without retraining a deep RL model live.

This notebook should support the thesis explanation of:

```text
current portfolio
+ selected strategy
+ latest known data/features
+ model/proxy/rule output
+ constraints
→ decision alternatives
→ risk output
→ audit entry
```

## Implementation

Create or prepare:

```text
research/notebooks/04_fast_layer_decision_case.ipynb
```

Recommended sections:

```text
1. Purpose
2. Example portfolio
3. Example strategy
4. Available market features
5. Decision alternatives
6. Risk output
7. Audit trace
8. Thesis figure/table export
```

## Acceptance Criteria

- Notebook file exists
- It clearly states its fast-layer purpose
- It demonstrates the no-live-retraining principle
- It can generate or mock a decision case
- It identifies what must later be connected to the system backend
- It exports thesis-ready evidence or placeholder outputs

## Notes

This notebook bridges research and the user-facing DSS.
""",
    },
    {
        "title": "Create fast-layer strategy constraints notebook skeleton",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "notebook", "fast-layer", "strategy", "decision"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Create a research notebook showing how user-defined strategy constraints affect decision output.

## Description

The strategy layer is central to the DSS. A custom strategy should not automatically trigger new deep RL training. Instead, the fast layer should apply constraints on top of available models/proxies/rules.

Examples:

- locked assets
- max position size
- risk profile
- stop-loss
- take-profit
- strategy switch allowed/not allowed

## Implementation

Create or prepare:

```text
research/notebooks/05_fast_layer_strategy_constraints.ipynb
```

Show at least one example where:

```text
same market state
+ different strategy constraints
→ different decision alternatives
```

## Acceptance Criteria

- Notebook file exists
- It explains constraints clearly
- It demonstrates at least one constraint effect
- It supports the thesis argument for human-in-the-loop decision support
- It does not require heavy RL training

## Notes

This supports both system design and methodology.
""",
    },
    {
        "title": "Create fast-layer audit reproducibility experiment",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "experiment", "fast-layer", "audit", "evaluation"],
        "category": CATEGORY_EVALUATION,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Create a small reproducibility experiment for the fast-layer audit trail.

## Description

The thesis needs to show that a generated recommendation can be traced back to the data, strategy, features and decision logic used at the time.

## Implementation

Create or prepare:

```text
research/experiments/fast_layer/audit_reproducibility/
```

The experiment should demonstrate:

```text
decision_time
strategy_id
portfolio_id
dataset_build_id
feature_build_id
model_or_rule_version
decision_output
risk_metrics
audit_log_id
```

## Acceptance Criteria

- Experiment folder exists
- Example audit input/output is documented
- Reproducibility fields are listed
- It is clear how this maps to the backend audit log
- It supports thesis evidence generation

## Notes

This is one of the most important trust/transparency tasks.
""",
    },
    {
        "title": "Define slow-to-fast model output contract",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["architecture", "contract", "integration", "slow-layer", "fast-layer", "model-registry"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "body": """## Goal

Define the contract between slow-layer model training outputs and fast-layer decision support.

## Description

The slow layer may produce trained models, baseline metrics, risk estimates or proxy outputs. The fast layer must know how to consume these outputs without needing to understand the entire training pipeline.

## Implementation

Define a simple contract for slow-layer outputs:

```json
{
  "model_id": "finrl_baseline_v001",
  "model_type": "finrl_baseline",
  "strategy_profile": "balanced",
  "trained_at": "2026-05-12T12:00:00Z",
  "data_snapshot_id": "snapshot_001",
  "feature_build_id": "features_001",
  "checkpoint_path": "...",
  "metrics_path": "...",
  "supported_assets": ["NVDA", "AAPL", "MSFT"],
  "risk_outputs": ["volatility", "drawdown", "cvar_proxy"],
  "wandb_run_id": "optional"
}
```

## Acceptance Criteria

- Contract is documented
- Required fields are separated from optional fields
- Fast layer can identify which model/proxy to use
- W&B run ID is included as optional metadata
- Contract can later map to DuckDB model registry
- Contract does not force V1.0 to implement full ML production infrastructure

## Notes

This keeps training and inference loosely coupled.
""",
    },
    {
        "title": "Define fast-layer inference input contract",
        "milestone": "M3 — Decision Support",
        "labels": ["architecture", "contract", "fast-layer", "decision", "strategy"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "body": """## Goal

Define the minimum input contract required by the fast-layer decision engine.

## Description

The fast layer needs a stable input structure that can combine user strategy, portfolio state, known market data and optional model outputs.

## Implementation

Define an input contract such as:

```json
{
  "decision_time": "2026-05-12T12:00:00Z",
  "user_id": "demo_user",
  "portfolio": {
    "cash": 10000,
    "positions": []
  },
  "strategy": {
    "risk_profile": "balanced",
    "locked_assets": ["NVDA"],
    "max_position_size": 0.25
  },
  "market_state": {
    "dataset_build_id": "snapshot_001",
    "feature_build_id": "features_001"
  },
  "model_context": {
    "model_id": "finrl_baseline_v001",
    "model_type": "baseline_or_proxy"
  }
}
```

## Acceptance Criteria

- Input contract is documented
- Contract includes decision time
- Contract includes strategy constraints
- Contract includes portfolio state
- Contract includes data/model references
- Contract is usable by backend and research notebooks

## Notes

This task supports both implementation and thesis methodology.
""",
    },
    {
        "title": "Define W&B logging convention for slow-layer experiments",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["wandb", "research", "slow-layer", "experiment", "evaluation"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Define how Weights & Biases should be used for slow-layer training and evaluation experiments.

## Description

The project should log ML/RL training and evaluation metadata consistently. This applies to research notebooks, reproducible experiments and later training workers.

## Implementation

Document a minimal convention:

```text
project: StockInvestmentDSS
entity: <configured locally>
run name: <experiment>_<date>_<short_id>
tags:
  - slow-layer
  - finrl
  - baseline
  - v1-poc
```

Log at least:

- config
- dataset snapshot ID
- feature build ID
- seed
- model type
- training duration
- metrics
- artifact/checkpoint path
- limitations

## Acceptance Criteria

- W&B usage convention is documented
- It is clear which experiments should log to W&B
- It is clear which values are safe to log
- No secrets/API keys are committed
- V1.0 can use a placeholder if W&B is not fully wired yet

## Notes

W&B should support reproducibility, not become a blocker.
""",
    },
    {
        "title": "Create slow-fast integration smoke test",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["integration", "slow-layer", "fast-layer", "evaluation", "test"],
        "category": CATEGORY_EVALUATION,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_POC,
        "deadline": "2026-05-14",
        "body": """## Goal

Create a smoke test proving that a slow-layer output can be referenced by a fast-layer decision.

## Description

This does not need to run full RL training. For V1.0, it is enough to use a baseline/proxy model output or a mocked model registry entry.

## Implementation

Test flow:

```text
1. Create or load model/proxy metadata
2. Store/reference it in DuckDB or a local JSON registry
3. Create a demo portfolio
4. Create a demo strategy
5. Run fast-layer decision engine
6. Verify decision output references model/proxy metadata
7. Verify audit log contains the reference
```

## Acceptance Criteria

- Smoke test exists
- Slow-layer artifact/model metadata is referenced
- Fast-layer decision output is generated
- Audit log contains model/proxy reference
- Test can be used as thesis evidence

## Notes

This is a key end-to-end architecture validation.
""",
    },
    {
        "title": "Document slow-fast mapping in README and research README",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["documentation", "research", "slow-layer", "fast-layer", "report"],
        "category": CATEGORY_REPORT,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-14",
        "body": """## Goal

Document the mapping between slow/fast architecture, research artifacts and system implementation.

## Description

The README already describes the slow/fast principle. This task ensures the structure also explains where each part lives.

## Implementation

Update relevant documentation with a mapping table:

| Layer | Research location | System location | Purpose |
|---|---|---|---|
| Slow | research/notebooks + research/experiments/slow_layer | workers/finrl-worker + training-job | training, backtesting, metrics |
| Fast | research/notebooks + research/experiments/fast_layer | backend + decision-worker | user-facing decision support |
| Shared | packages/ | packages/ | schemas, risk, data, strategy |

## Acceptance Criteria

- README or research README includes slow/fast mapping
- It is clear what belongs in notebooks
- It is clear what belongs in experiments
- It is clear what belongs in the runnable system
- The explanation supports the thesis system design section

## Notes

Keep this concise and practical.
""",
    },
    {
        "title": "Freeze V1.0 slow-fast layer scope",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["planning", "documentation", "slow-layer", "fast-layer"],
        "category": CATEGORY_REPORT,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_POC,
        "deadline": "2026-05-14",
        "body": """## Goal

Freeze what slow-layer and fast-layer functionality belongs in V1.0.

## Description

The slow/fast architecture can easily expand into a large ML platform. For V1.0, the scope must remain controlled.

## Implementation

Write a short V1.0 scope note:

```text
Included in V1.0:
- simple slow-layer baseline/proxy output
- basic model/proxy metadata
- fast-layer decision case
- risk output
- audit reference
- thesis evidence

Not included in V1.0:
- exhaustive hyperparameter optimization
- production-grade model serving
- full live trading integration
- full IQN/evidential model if time does not allow it
```

## Acceptance Criteria

- Scope is documented
- Included and excluded items are explicit
- The scope supports the thesis narrative
- Advanced work is marked as future work where appropriate

## Notes

This prevents backlog expansion before the PoC is working.
""",
    },
]


def run(cmd: List[str], check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        sys.exit(result.returncode)
    return (result.stdout or "").strip()


def gh_json(cmd: List[str]) -> Any:
    out = run(cmd)
    return json.loads(out) if out else {}


def graphql_rate_remaining() -> Optional[int]:
    try:
        data = gh_json([
            "gh", "api", "graphql",
            "-f", "query={ rateLimit { remaining used resetAt } }",
        ])
        rl = data.get("data", {}).get("rateLimit", {})
        print("")
        print("GraphQL rate limit:")
        print(f"  remaining: {rl.get('remaining')}")
        print(f"  used:      {rl.get('used')}")
        print(f"  resetAt:   {rl.get('resetAt')}")
        return rl.get("remaining")
    except Exception as exc:
        print(f"WARNING: Could not read GraphQL rate limit: {exc}")
        return None


def issue_exists(title: str) -> Optional[int]:
    data = gh_json([
        "gh", "issue", "list",
        "--repo", GH_REPO,
        "--state", "all",
        "--search", f"{title} in:title",
        "--json", "number,title",
        "--limit", "100",
    ])
    for item in data:
        if item.get("title") == title:
            return int(item["number"])
    return None


def create_or_find_issue(task: Dict[str, Any]) -> int:
    title = task["title"]
    existing = issue_exists(title)
    if existing:
        print(f"Issue already exists: {title} (#{existing})")
        return existing

    cmd = [
        "gh", "issue", "create",
        "--repo", GH_REPO,
        "--title", title,
        "--body", task["body"],
    ]

    labels = task.get("labels") or []
    if labels:
        cmd.extend(["--label", ",".join(labels)])

    milestone = task.get("milestone")
    if milestone:
        cmd.extend(["--milestone", milestone])

    print(f"Creating issue: {title}")
    url = run(cmd)
    number = int(url.rstrip("/").split("/")[-1])
    print(f"  Created #{number}")
    return number


def get_issue_url(number: int) -> str:
    return run([
        "gh", "issue", "view", str(number),
        "--repo", GH_REPO,
        "--json", "url",
        "--jq", ".url",
    ])


def add_issue_to_project(number: int) -> None:
    url = get_issue_url(number)
    result = subprocess.run(
        ["gh", "project", "item-add", PROJECT_NUMBER, "--owner", OWNER, "--url", url],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    if result.returncode == 0:
        print("  Added/found in project.")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        if "already exists" in msg.lower():
            print("  Already in project.")
        else:
            print(f"  WARNING: Could not add to project: {msg}")


def fetch_project_metadata() -> Dict[str, Any]:
    project = gh_json([
        "gh", "project", "view", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
    ])
    fields_data = gh_json([
        "gh", "project", "field-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
        "--limit", "100",
    ])
    return {
        "project_id": project["id"],
        "fields": {field["name"]: field for field in fields_data["fields"]},
    }


def fetch_project_items() -> Dict[str, str]:
    data = gh_json([
        "gh", "project", "item-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
        "--limit", "300",
    ])

    by_title: Dict[str, str] = {}
    for item in data.get("items", []):
        content = item.get("content") or {}
        title = content.get("title")
        if title:
            by_title[title] = item["id"]
    return by_title


def set_single_select(project_id: str, fields: Dict[str, Any], item_id: str, field_name: str, option_name: str) -> None:
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return

    option_id = None
    for option in field.get("options", []):
        if option.get("name") == option_name:
            option_id = option["id"]
            break

    if not option_id:
        print(f"  WARNING: Option not found: {field_name} -> {option_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--single-select-option-id", option_id,
    ])
    print(f"  Set {field_name} = {option_name}")


def set_date(project_id: str, fields: Dict[str, Any], item_id: str, field_name: str, value: str) -> None:
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--date", value,
    ])
    print(f"  Set {field_name} = {value}")


def set_number(project_id: str, fields: Dict[str, Any], item_id: str, field_name: str, value: int) -> None:
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return

    run([
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field["id"],
        "--number", str(value),
    ])
    print(f"  Set {field_name} = {value}")


graphql_rate_remaining()

print("")
print("Creating/finding slow-fast alignment issues...")
issue_numbers = []
for task in TASKS:
    number = create_or_find_issue(task)
    issue_numbers.append(number)
    add_issue_to_project(number)

print("")
print("Fetching project metadata...")
meta = fetch_project_metadata()
project_id = meta["project_id"]
fields = meta["fields"]

print("Fetching project items...")
items_by_title = fetch_project_items()

print("")
print("Setting project fields...")
for task in TASKS:
    title = task["title"]
    item_id = items_by_title.get(title)

    if not item_id:
        print(f"WARNING: Could not find project item for: {title}")
        continue

    print(f"Updating fields: {title}")

    set_single_select(project_id, fields, item_id, "Status", task["status"] if "status" in task else STATUS_TODO)
    set_single_select(project_id, fields, item_id, "Category", task["category"])
    set_single_select(project_id, fields, item_id, "Priority", task["priority"])
    set_single_select(project_id, fields, item_id, "Roadmap", task["roadmap"])
    set_single_select(project_id, fields, item_id, "Track", task["track"])
    set_single_select(project_id, fields, item_id, "Percentage", PERCENT_0)
    set_date(project_id, fields, item_id, "Deadline", task["deadline"])
    set_number(project_id, fields, item_id, "Progress Number", 0)

print("")
graphql_rate_remaining()

print("")
print("Done.")
print("")
print("Script 13 completed.")
print("")
print("Created/found tasks:")
for number, task in zip(issue_numbers, TASKS):
    print(f"- #{number} {task['title']}")

PY
