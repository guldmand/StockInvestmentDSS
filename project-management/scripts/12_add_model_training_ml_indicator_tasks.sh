#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 12
# Add model training, hyperparameter, ML indicators and W&B tasks
# ============================================================
#
# Purpose:
# - Adds the missing V1.0 PoC tasks around ML/Deep RL training protocol,
#   hyperparameter scope, ML-derived indicators and experiment tracking.
# - Keeps V1.0 sharply scoped: one baseline/proxy training run is enough.
# - Adds Weights & Biases logging tasks for both research and application/slow-layer runs.
# - Sets project fields: Status, Category, Priority, Roadmap, Track, Percentage,
#   Deadline and Progress Number.
#
# Safe to re-run:
# - Existing labels are reused.
# - Existing issues with identical title are reused.
# - Project fields are updated idempotently.
#
# Run from repository root:
#   & "C:\Program Files\Git\bin\bash.exe" ".\project-management\scripts\12_add_model_training_ml_indicator_tasks.sh"
#
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

# Project field option names. These must match the GitHub Project fields.
STATUS_TODO="Todo"
STATUS_IN_PROGRESS="In Progress"
STATUS_CODE_REVIEW="Code Review"
STATUS_DONE="Done"

CATEGORY_DEVELOPMENT="⚙️ Development"
CATEGORY_DATA="📊 Data"
CATEGORY_RESEARCH="📚 Research"
CATEGORY_ARCHITECTURE="🏗️ Architecture"
CATEGORY_RL_AI="🤖 RL / AI"
CATEGORY_EVALUATION="🧪 Evaluation"
CATEGORY_REPORT="📝 Report"

PRIORITY_URGENT="🗼 Urgent"
PRIORITY_HIGH="⛰️ High"
PRIORITY_MEDIUM="🫣 Medium"
PRIORITY_LOW="🌈 Low"

ROADMAP_NOW="✅ Now"
ROADMAP_NEXT="🔜 Next"
ROADMAP_LATER="🗓️ Later"

TRACK_POC="PoC"
TRACK_REPORT="Report"
TRACK_INFRA="Infra"
TRACK_RESEARCH="Research"
TRACK_BONUS="Bonus"
TRACK_FUTURE="Future Work"

PERCENTAGE_ZERO="□□□□□□□□□□ 0%"

# Milestones from script 1.
MILESTONE_M2="M2 — Strategy + Portfolio"
MILESTONE_M3="M3 — Decision Support"
MILESTONE_M4="M4 — Risk, Uncertainty, Audit"
MILESTONE_M5="M5 — Demo + Report Figures"
MILESTONE_M6="M6 — Buffer / v1.1"

# Labels used by these tasks. Keep labels short and clean; priority is a Project field, not a label.
ensure_label() {
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

echo "Using repo: $GH_REPO"
echo "Using project: $PROJECT_TITLE (#$PROJECT_NUMBER)"
echo ""

echo "Checking GitHub auth..."
gh auth status

echo ""
echo "Creating required labels..."
ensure_label "ml" "5319E7" "Machine learning models, indicators and training logic"
ensure_label "deeprl" "FBCA04" "Deep reinforcement learning experiments and training"
ensure_label "hyperparameters" "F9D0C4" "Hyperparameter configs, tuning scope and limitations"
ensure_label "wandb" "1D76DB" "Weights & Biases experiment tracking"
ensure_label "experiment-tracking" "0E8A16" "Experiment logs, metadata and reproducibility tracking"
ensure_label "model-registry" "006B75" "Model checkpoint and registry metadata"
ensure_label "checkpoint" "C5DEF5" "Model checkpoint storage and artifact handling"
ensure_label "indicators" "D4C5F9" "Technical and model-derived indicators"
ensure_label "slow-layer" "5319E7" "Offline training, evaluation and scheduled jobs"
ensure_label "fast-layer" "BFD4F2" "Online decision support and inference"
ensure_label "research" "0E8A16" "Academic notebooks, experiments and report-facing work"
ensure_label "rl" "FBCA04" "Reinforcement learning, FinRL, Gymnasium and agents"
ensure_label "finrl" "FBCA04" "FinRL framework integration"
ensure_label "risk" "B60205" "Risk, uncertainty, CVaR and downside metrics"
ensure_label "evaluation" "5319E7" "Backtesting, metrics and evaluation"
ensure_label "report" "0E8A16" "Thesis report writing and report artifacts"
ensure_label "documentation" "0075CA" "README, docs, runbooks and project notes"
ensure_label "data" "C5DEF5" "Data pipeline, storage, DuckDB and features"
ensure_label "architecture" "7057FF" "Architecture and system design"

echo ""
echo "Checking GraphQL rate limit before Project updates..."
gh api graphql -f query='{ rateLimit { limit cost remaining used resetAt } }' \
  --jq '.data.rateLimit | "  remaining: \(.remaining)\n  used:      \(.used)\n  resetAt:   \(.resetAt)"' || true

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
PROJECT_TITLE = "StockInvestmentDSS PoC Sprint"

STATUS_TODO = "Todo"
CATEGORY_RL_AI = "🤖 RL / AI"
CATEGORY_RESEARCH = "📚 Research"
CATEGORY_DATA = "📊 Data"
CATEGORY_ARCHITECTURE = "🏗️ Architecture"
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
TRACK_FUTURE = "Future Work"
TRACK_REPORT = "Report"

PERCENTAGE_ZERO = "□□□□□□□□□□ 0%"

MILESTONE_M3 = "M3 — Decision Support"
MILESTONE_M4 = "M4 — Risk, Uncertainty, Audit"
MILESTONE_M5 = "M5 — Demo + Report Figures"
MILESTONE_M6 = "M6 — Buffer / v1.1"

TASKS: List[Dict[str, Any]] = [
    {
        "title": "Define model training protocol",
        "milestone": MILESTONE_M4,
        "labels": ["rl", "ml", "deeprl", "training", "documentation", "slow-layer"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Define the minimal model training protocol for the V1.0 PoC.

The protocol must explain how training runs are started, configured, logged, evaluated and connected back to the decision-support system.

## Description

This is not a task to build a full training platform. It is a task to write down the rules for how V1.0 training experiments should be run so that the thesis can describe them honestly and reproducibly.

The protocol should cover both tracks:

1. Research track: notebooks and structured experiments under `research/`.
2. Application/slow-layer track: training jobs or workers under `system/` that can later serve the DSS.

The V1.0 goal is to demonstrate the training pipeline and one small baseline/proxy training run, not to perform exhaustive optimization.

## Acceptance Criteria

- A short protocol document exists, for example `docs/thesis/model-training-protocol.md` or `research/README.md` section.
- Protocol describes input data, config files, random seeds, output artifacts and evaluation metrics.
- Protocol explains the split between research notebooks and application slow-layer jobs.
- Protocol defines where model checkpoints and metrics are stored.
- Protocol mentions Weights & Biases logging for experiment traceability.
- Protocol clearly states what is inside and outside V1.0 scope.

## Initial AI Agent Instruction

Read `README.md`, the implementation overview notes and the current project structure before editing.

Create a concise model training protocol that supports the thesis narrative. Do not over-engineer. Focus on a minimal reproducible pipeline for one small FinRL/Deep RL baseline or proxy run.

Make sure the document distinguishes clearly between:

- exploratory research notebooks
- structured experiments
- slow-layer application training jobs
- fast-layer online decision support

## Notes

Strong thesis formulation:

> Due to time and compute constraints, extensive hyperparameter optimization is outside the V1.0 PoC scope. The system architecture supports later scheduled tuning through the slow-layer training pipeline.
""",
    },
    {
        "title": "Define hyperparameter tuning scope for V1.0",
        "milestone": MILESTONE_M4,
        "labels": ["ml", "deeprl", "hyperparameters", "documentation", "research"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Define what hyperparameter tuning means for V1.0 and what is explicitly out of scope.

## Description

The project should not get blocked by trying to perform exhaustive tuning. For the V1.0 PoC, tuning should be limited to a small, explainable baseline configuration.

This task exists to protect the project scope and to make the report honest.

## Acceptance Criteria

- A short tuning-scope document exists.
- V1.0 tuning scope is limited to a small baseline config or a few manually chosen values.
- Exhaustive grid search, Bayesian optimization and large-scale sweeps are marked as future work.
- The document explains why the limitation is reasonable for a thesis PoC.
- The document references compute/time constraints and slow-layer extensibility.
- The text can be reused in the methodology or limitations section.

## Initial AI Agent Instruction

Write a scope note that prevents the project from becoming a hyperparameter tuning project.

The correct V1.0 position is:

- run one small training experiment or proxy
- store metrics and checkpoint
- document limitations
- prepare architecture for future tuning

Do not propose large sweeps for V1.0.

## Notes

The task is mainly a thesis-quality scoping task, not a heavy implementation task.
""",
    },
    {
        "title": "Create baseline hyperparameter config",
        "milestone": MILESTONE_M4,
        "labels": ["ml", "deeprl", "hyperparameters", "finrl", "research"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Create a small baseline hyperparameter configuration for the first FinRL/Deep RL experiment.

## Description

The purpose is to have one reproducible config that can be used by a notebook or experiment runner.

This config should be intentionally modest. It should allow the system to demonstrate the training/evaluation/checkpoint pipeline without requiring days of compute.

## Suggested location

```text
research/configs/experiment_005_baseline_training.yaml
```

or, if the application slow-layer consumes it:

```text
system/configs/training_baseline.yaml
```

## Acceptance Criteria

- A baseline training config exists.
- Config includes environment/data parameters.
- Config includes algorithm/model name.
- Config includes basic training parameters such as seed, timesteps/episodes, learning rate and evaluation window.
- Config includes output paths for metrics, checkpoints and W&B run metadata.
- Config is small enough to run as a PoC experiment.

## Initial AI Agent Instruction

Create a minimal YAML config for a first baseline RL/FinRL training run.

Prefer clarity over completeness. Add comments where useful.

Do not invent a large hyperparameter sweep. This is a baseline config for a controlled PoC run.

Include W&B fields such as:

```yaml
tracking:
  enabled: true
  provider: wandb
  project: stockinvestmentdss
  entity: null
  tags: [v1-poc, baseline, finrl]
```

## Notes

This config should make later automation possible without forcing it immediately.
""",
    },
    {
        "title": "Run small FinRL training experiment",
        "milestone": MILESTONE_M4,
        "labels": ["finrl", "rl", "ml", "deeprl", "training", "research", "wandb"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Run one small FinRL training experiment or equivalent proxy experiment for the V1.0 PoC.

## Description

The purpose is to demonstrate that the training pipeline can execute, log metrics and produce an artifact. The output does not need to be a state-of-the-art model.

The run should be small enough to complete on available hardware or a modest cloud/GPU setup.

## Acceptance Criteria

- A small training run completes or a documented proxy run is produced.
- Run uses a reproducible config.
- Metrics are exported to file and/or DuckDB.
- A model checkpoint or placeholder artifact is stored.
- The run is logged to Weights & Biases if credentials are available.
- Limitations are documented.
- Output can be referenced in the thesis as PoC evidence.

## Initial AI Agent Instruction

Implement the smallest useful FinRL/Deep RL training run that supports the thesis.

Do not optimize for performance. Optimize for reproducibility, traceability and successful end-to-end execution.

Use W&B logging if `WANDB_API_KEY` or a local authenticated W&B session is available. If not, make logging optional and do not fail the run.

## Notes

A weak but reproducible baseline is better than an ambitious run that never finishes.
""",
    },
    {
        "title": "Store model checkpoint and metrics",
        "milestone": MILESTONE_M4,
        "labels": ["model-registry", "checkpoint", "ml", "duckdb", "wandb", "evaluation"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Store model checkpoint metadata and training/evaluation metrics in a reproducible way.

## Description

Training outputs must not disappear into notebooks. The PoC needs a clear artifact path and metadata trail.

This task connects training results to:

- filesystem / guldNAS artifact storage
- DuckDB model registry or metrics tables
- Weights & Biases run metadata
- thesis result exports

## Acceptance Criteria

- Checkpoint output path is defined.
- Metrics output path is defined.
- At least one metrics file is written as CSV, Parquet or JSON.
- DuckDB model registry schema is used or prepared.
- W&B run URL or run ID can be stored if W&B is enabled.
- README or protocol explains where artifacts live.

## Initial AI Agent Instruction

Add minimal artifact storage logic for model checkpoints and metrics.

Do not commit large checkpoint files to Git. Use `.gitkeep` and README notes for artifact folders.

Support local development paths and guldNAS paths.

## Notes

This task supports traceability and reproducibility more than model performance.
""",
    },
    {
        "title": "Document hyperparameter limitations",
        "milestone": MILESTONE_M5,
        "labels": ["hyperparameters", "documentation", "report", "research"],
        "category": CATEGORY_REPORT,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_REPORT,
        "deadline": "2026-05-14",
        "body": """## Goal

Document the limitations of V1.0 hyperparameter tuning for the thesis report.

## Description

This task ensures the report is honest about what was and was not optimized.

The goal is to frame limited tuning as a controlled scope decision, not as a hidden weakness.

## Acceptance Criteria

- A short limitations text exists in report notes or the discussion section.
- Text explains that exhaustive tuning is outside V1.0 scope.
- Text explains that architecture supports later slow-layer scheduled tuning.
- Text mentions compute and time constraints.
- Text does not overclaim model performance.

## Initial AI Agent Instruction

Write a clear, academically honest limitation section.

Use this core position:

> V1.0 demonstrates the architecture, data flow, decision pipeline and evaluation structure. Extensive hyperparameter optimization is treated as future work.

Do not make claims that the baseline model is optimal.

## Notes

This text can be reused in Methodology, Discussion or Future Work.
""",
    },
    {
        "title": "Create hyperparameter tuning future-work plan",
        "milestone": MILESTONE_M6,
        "labels": ["hyperparameters", "future-work", "ml", "deeprl", "documentation"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_LOW,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_FUTURE,
        "deadline": "2026-05-17",
        "body": """## Goal

Create a future-work plan for systematic hyperparameter tuning after V1.0.

## Description

The V1.0 PoC should not perform exhaustive tuning, but the architecture should show how tuning could be added later.

This task documents the future extension path.

## Acceptance Criteria

- Future-work note exists.
- Plan mentions scheduled slow-layer tuning.
- Plan mentions W&B sweeps or similar experiment tracking.
- Plan mentions compute targets such as local GPU box, Vast.ai, Colab or Azure.
- Plan explains how tuned models would enter the model registry.

## Initial AI Agent Instruction

Write a concise future-work plan. Do not implement tuning now.

Focus on how the system could later support:

- W&B sweeps
- scheduled training jobs
- model comparison
- registry promotion
- rollback to previous model

## Notes

This is useful for thesis future work and architectural credibility.
""",
    },
    {
        "title": "Define ML-derived indicator schema",
        "milestone": MILESTONE_M4,
        "labels": ["ml", "indicators", "data", "risk", "decision"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "body": """## Goal

Define a schema for ML-derived indicators that can later be consumed by the decision engine.

## Description

The system already has technical indicators. This task defines a separate concept for indicators produced by models.

Examples:

- predicted return class
- volatility regime
- uncertainty score
- downside-risk score
- model confidence
- strategy suitability score

The goal is to make the architecture ready without requiring all indicators to be implemented in V1.0.

## Acceptance Criteria

- A schema or table design for ML-derived indicators exists.
- Schema distinguishes technical indicators from model-derived indicators.
- Schema includes timestamp / point-in-time metadata.
- Schema includes model ID or indicator source.
- Schema includes confidence/uncertainty fields where relevant.
- Schema can be stored in DuckDB or exported as Parquet/CSV.

## Initial AI Agent Instruction

Define the smallest useful ML-derived indicator schema.

Do not implement advanced models. Focus on how output from a model would be represented and connected to the DSS.

Ensure point-in-time correctness: an indicator must have a known timestamp and model/source metadata.

## Notes

This task supports both the application and the thesis system-design chapter.
""",
    },
    {
        "title": "Create ML indicator placeholder pipeline",
        "milestone": MILESTONE_M4,
        "labels": ["ml", "indicators", "data", "features", "slow-layer"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_POC,
        "deadline": "2026-05-13",
        "body": """## Goal

Create a placeholder pipeline for ML-derived indicators.

## Description

The placeholder pipeline should make it possible to generate, store and consume simple model-derived indicators without requiring a complete model stack.

For V1.0, this can be rule-based or dummy/proxy output as long as it follows the real schema.

## Acceptance Criteria

- Placeholder pipeline function or script exists.
- Pipeline writes output using the ML-derived indicator schema.
- Output can be stored in DuckDB, CSV and/or Parquet.
- Output includes model/source metadata.
- Pipeline is documented as placeholder/proxy if not model-based yet.

## Initial AI Agent Instruction

Implement a minimal placeholder pipeline that produces a few ML-derived indicator rows.

Keep it simple and transparent. The main value is that the rest of the DSS can consume the indicator shape later.

Do not pretend placeholder values are real predictions.

## Notes

This helps prevent the thesis architecture from being purely theoretical.
""",
    },
    {
        "title": "Document technical indicators versus ML-derived indicators",
        "milestone": MILESTONE_M5,
        "labels": ["documentation", "indicators", "ml", "features", "report"],
        "category": CATEGORY_REPORT,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_REPORT,
        "deadline": "2026-05-14",
        "body": """## Goal

Document the difference between technical indicators and ML-derived indicators.

## Description

The report and README should not blur feature engineering, model output and decision logic.

Technical indicators are computed directly from market data. ML-derived indicators are outputs from trained or proxy models.

## Acceptance Criteria

- A short explanation exists in docs or report notes.
- Technical indicators and ML-derived indicators are defined separately.
- Examples are provided for each.
- The document explains how both can feed the decision engine.
- The document explains that V1.0 may use placeholders for some ML-derived indicators.

## Initial AI Agent Instruction

Write a clear distinction suitable for the System Design or Methodology section.

Make the terminology consistent with README.md.

Do not overclaim that all ML indicators are implemented if they are placeholders.

## Notes

This improves thesis clarity and reduces implementation confusion.
""",
    },
    {
        "title": "Add Weights and Biases experiment tracking setup",
        "milestone": MILESTONE_M4,
        "labels": ["wandb", "experiment-tracking", "ml", "deeprl", "research", "slow-layer"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": """## Goal

Add basic Weights & Biases tracking support for ML, Deep RL, FinRL and hyperparameter-related experiments.

## Description

The project should log model training and experiment results to Weights & Biases when available.

This applies to both:

1. research notebooks / research experiments
2. application slow-layer training jobs

W&B must be optional so local tests do not fail if credentials are missing.

## Acceptance Criteria

- W&B is added to relevant requirements if needed.
- `.env.example` documents W&B variables.
- Training scripts can enable/disable W&B logging.
- W&B project name is documented.
- W&B run ID or URL can be stored in metrics/model registry metadata.
- Missing W&B credentials do not break local PoC execution.

## Suggested environment variables

```text
WANDB_MODE=online
WANDB_PROJECT=stockinvestmentdss
WANDB_ENTITY=
WANDB_API_KEY=
WANDB_TAGS=v1-poc,finrl,baseline
```

For offline/dev mode:

```text
WANDB_MODE=offline
```

## Initial AI Agent Instruction

Add optional W&B tracking in the simplest possible way.

Do not require W&B for basic local startup or tests. The app must still run without W&B credentials.

Use environment variables and document them in `.env.example`.

## Notes

This is important for reproducibility and thesis evidence.
""",
    },
    {
        "title": "Create W&B logging helper for research experiments",
        "milestone": MILESTONE_M4,
        "labels": ["wandb", "experiment-tracking", "research", "ml", "deeprl"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Create a small reusable W&B logging helper for research notebooks and experiments.

## Description

The helper should avoid copy-pasting W&B setup code across notebooks.

It should support online, offline and disabled modes.

## Acceptance Criteria

- Helper exists under `research/` or a shared package.
- Helper initializes W&B from config/environment variables.
- Helper logs config and metrics.
- Helper returns run ID/URL when available.
- Helper fails gracefully if W&B is not configured.

## Initial AI Agent Instruction

Create a minimal helper. Do not build a full experiment platform.

The helper should be easy to use from notebooks and scripts:

```python
run = init_tracking(config)
log_metrics({"sharpe": value})
finish_tracking()
```

Keep the implementation simple and optional.

## Notes

This supports reproducibility without slowing down development.
""",
    },
    {
        "title": "Connect W&B run metadata to model registry",
        "milestone": MILESTONE_M4,
        "labels": ["wandb", "model-registry", "duckdb", "checkpoint", "experiment-tracking"],
        "category": CATEGORY_RL_AI,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": """## Goal

Connect Weights & Biases run metadata to the model registry / experiment metadata.

## Description

If a training run is logged in W&B, the DSS should be able to trace a stored checkpoint or metrics row back to the W&B run.

## Acceptance Criteria

- Model registry schema includes optional W&B fields.
- Suggested fields include `wandb_project`, `wandb_run_id`, `wandb_run_url`, `wandb_tags`.
- Training output stores W&B metadata when available.
- Missing W&B metadata is allowed.
- Documentation explains how W&B and DuckDB registry relate.

## Initial AI Agent Instruction

Extend the model registry / experiment metadata design minimally.

Do not require W&B to exist for every run. Treat W&B as optional but useful traceability metadata.

## Notes

This strengthens auditability and experiment reproducibility.
""",
    },
    {
        "title": "Define model checkpoint storage policy",
        "milestone": MILESTONE_M4,
        "labels": ["checkpoint", "model-registry", "guldnas", "documentation", "ml"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_POC,
        "deadline": "2026-05-12",
        "body": """## Goal

Define where model checkpoints are stored and how they are referenced.

## Description

Model checkpoints should not be committed to Git. They should live in runtime storage, preferably guldNAS for persistent storage.

The repository should contain only documentation, folder placeholders and metadata schemas.

## Acceptance Criteria

- Checkpoint storage policy exists.
- Policy distinguishes local development path and guldNAS path.
- Policy says large checkpoints are ignored by Git.
- Policy defines how checkpoints are referenced from DuckDB/model registry.
- Policy includes relation to W&B artifacts if used.

## Suggested canonical path

```text
/mnt/nas/stockinvestmentdss/model-checkpoints/
├── finrl/
├── iqn/
├── baselines/
└── metadata/
```

## Initial AI Agent Instruction

Write a small storage policy document.

Do not add large binary artifacts. Add `.gitkeep` and README files only if needed.

Connect this policy to the canonical storage layout in README.md.

## Notes

This supports both application deployment and thesis reproducibility.
""",
    },
    {
        "title": "Define model registry update workflow",
        "milestone": MILESTONE_M4,
        "labels": ["model-registry", "duckdb", "ml", "checkpoint", "slow-layer"],
        "category": CATEGORY_ARCHITECTURE,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_POC,
        "deadline": "2026-05-14",
        "body": """## Goal

Define how trained/proxy models are registered after training.

## Description

The slow layer should eventually be able to train models, evaluate them and register metadata so the fast layer can select an available model.

V1.0 only needs a simple workflow description and optionally a minimal registry insert script.

## Acceptance Criteria

- Workflow describes train -> evaluate -> store checkpoint -> store metrics -> register model.
- Workflow includes optional W&B run metadata.
- Workflow includes model status such as candidate/active/archived if useful.
- Workflow explains how fast-layer model selection can use the registry later.
- Workflow is consistent with DuckDB schema.

## Initial AI Agent Instruction

Create a minimal model registry workflow. Do not build a complex MLOps platform.

Focus on traceability:

- what model was trained
- with what config
- on what data snapshot
- what metrics it achieved
- where checkpoint is stored
- whether W&B run metadata exists

## Notes

This is the bridge between research experiments and the operational DSS.
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
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()


def gh_json(cmd: List[str]) -> Any:
    out = run(cmd)
    return json.loads(out) if out else {}


def issue_exists(title: str) -> Optional[int]:
    data = gh_json([
        "gh", "issue", "list",
        "--repo", GH_REPO,
        "--state", "all",
        "--limit", "300",
        "--json", "number,title",
        "--search", f'"{title}" in:title',
    ])
    for issue in data:
        if issue.get("title") == title:
            return int(issue["number"])
    return None


def create_or_get_issue(task: Dict[str, Any]) -> int:
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
        "--milestone", task["milestone"],
    ]
    labels = task.get("labels", [])
    if labels:
        cmd.extend(["--label", ",".join(labels)])

    url = run(cmd)
    number = int(url.rstrip("/").split("/")[-1])
    print(f"Created issue: {title} (#{number})")
    return number


def get_issue_url(number: int) -> str:
    return gh_json([
        "gh", "issue", "view", str(number),
        "--repo", GH_REPO,
        "--json", "url",
    ])["url"]


def get_project_id() -> str:
    data = gh_json([
        "gh", "project", "view", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
    ])
    return data["id"]


def get_fields() -> Dict[str, Dict[str, Any]]:
    data = gh_json([
        "gh", "project", "field-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
        "--limit", "100",
    ])
    return {field["name"]: field for field in data.get("fields", [])}


def get_items() -> List[Dict[str, Any]]:
    data = gh_json([
        "gh", "project", "item-list", PROJECT_NUMBER,
        "--owner", OWNER,
        "--format", "json",
        "--limit", "300",
    ])
    return data.get("items", [])


def add_issue_to_project(issue_url: str) -> None:
    run([
        "gh", "project", "item-add", PROJECT_NUMBER,
        "--owner", OWNER,
        "--url", issue_url,
    ], check=False)


def find_item_id_by_title(items: List[Dict[str, Any]], title: str) -> Optional[str]:
    for item in items:
        content = item.get("content") or {}
        if content.get("title") == title:
            return item.get("id")
    return None


def set_single_select(project_id: str, item_id: str, fields: Dict[str, Dict[str, Any]], field_name: str, option_name: str) -> None:
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return
    option_id = None
    for option in field.get("options", []):
        if option.get("name") == option_name:
            option_id = option.get("id")
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


def set_date(project_id: str, item_id: str, fields: Dict[str, Dict[str, Any]], field_name: str, value: str) -> None:
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


def set_number(project_id: str, item_id: str, fields: Dict[str, Dict[str, Any]], field_name: str, value: int) -> None:
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


print("Creating/finding issues...")
issue_numbers: Dict[str, int] = {}
for task in TASKS:
    issue_numbers[task["title"]] = create_or_get_issue(task)

print("")
print("Adding issues to project...")
for task in TASKS:
    number = issue_numbers[task["title"]]
    url = get_issue_url(number)
    add_issue_to_project(url)
    print(f"Added/found in project: #{number} — {task['title']}")

# Give GitHub a moment to reflect item-add operations.
time.sleep(2)

print("")
print("Fetching project metadata...")
project_id = get_project_id()
fields = get_fields()
items = get_items()

print("")
print("Setting project fields...")
for task in TASKS:
    title = task["title"]
    print(f"Updating fields: {title}")
    item_id = find_item_id_by_title(items, title)
    if not item_id:
        # Refresh once in case project item list was stale.
        items = get_items()
        item_id = find_item_id_by_title(items, title)
    if not item_id:
        print(f"  WARNING: Could not find project item for {title}")
        continue

    set_single_select(project_id, item_id, fields, "Status", STATUS_TODO)
    set_single_select(project_id, item_id, fields, "Category", task["category"])
    set_single_select(project_id, item_id, fields, "Priority", task["priority"])
    set_single_select(project_id, item_id, fields, "Roadmap", task["roadmap"])
    set_single_select(project_id, item_id, fields, "Track", task["track"])
    set_single_select(project_id, item_id, fields, "Percentage", PERCENTAGE_ZERO)
    set_date(project_id, item_id, fields, "Deadline", task["deadline"])
    set_number(project_id, item_id, fields, "Progress Number", 0)

print("")
print("Done.")
print("")
print("Script 12 completed.")
print("")
print("Created/updated task group:")
print("- model training protocol")
print("- hyperparameter tuning scope")
print("- baseline training config/run")
print("- ML-derived indicator schema/pipeline")
print("- model checkpoint and registry workflow")
print("- Weights & Biases tracking setup")
PY

echo ""
echo "Script 12 completed."
echo ""
echo "Next recommendation: keep these mostly in Next/Later. Only move W&B/model protocol tasks to Now when today's core PoC skeleton is stable."
