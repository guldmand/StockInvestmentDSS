#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 11
# Add external dependency + research notebook/experiment tasks
# Safe to run multiple times.
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

printf 'Using repo: %s\n' "$GH_REPO"
printf 'Using project: %s (#%s)\n\n' "$PROJECT_TITLE" "$PROJECT_NUMBER"

echo "Checking GitHub auth..."
gh auth status

PYTHON_BIN="python"
if ! command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"

PYTHONIOENCODING=utf-8 "$PYTHON_BIN" <<'PY'
import json
import subprocess
import sys

OWNER = "guldmand"
REPO = "StockInvestmentDSS"
GH_REPO = f"{OWNER}/{REPO}"
PROJECT_NUMBER = "11"

STATUS_TODO = "Todo"
PERCENT_0 = "□□□□□□□□□□ 0%"

PRIORITY_URGENT = "🗼 Urgent"
PRIORITY_HIGH = "⛰️ High"
PRIORITY_MEDIUM = "🫣 Medium"
PRIORITY_LOW = "🌈 Low"

ROADMAP_NOW = "✅ Now"
ROADMAP_NEXT = "🔜 Next"
ROADMAP_LATER = "🗓️ Later"

CATEGORY_DEV = "⚙️ Development"
CATEGORY_DATA = "📊 Data"
CATEGORY_RESEARCH = "📚 Research"
CATEGORY_ARCH = "🏗️ Architecture"
CATEGORY_RL = "🤖 RL / AI"
CATEGORY_EVAL = "🧪 Evaluation"
CATEGORY_REPORT = "📝 Report"

TRACK_POC = "PoC"
TRACK_RESEARCH = "Research"
TRACK_REPORT = "Report"
TRACK_FUTURE = "Future Work"

LABELS = {
    "external": ("5319E7", "External repositories and pinned dependency references"),
    "dependency": ("0366D6", "Dependency and integration management"),
    "finrl": ("F1C40F", "FinRL framework integration"),
    "gymnasium": ("7057FF", "Gymnasium environment integration"),
    "objectrl": ("9B59B6", "ObjectRL reference/prototyping integration"),
    "zero-sum": ("1D76DB", "Zero Sum frontend/charting reference"),
    "research": ("0E8A16", "Research notebooks and experiments"),
    "notebook": ("C5DEF5", "Jupyter notebook work"),
    "experiment": ("FBCA04", "Reproducible experiment work"),
    "reproducibility": ("BFDADC", "Reproducibility and repeatability"),
    "hyperparameter": ("D4C5F9", "Hyperparameter tuning"),
    "model-selection": ("D93F0B", "Model selection and comparison"),
    "ml": ("5319E7", "Machine learning models and features"),
    "features": ("C5DEF5", "Feature engineering"),
    "documentation": ("0075CA", "Documentation"),
    "devops": ("5319E7", "DevOps and automation"),
    "docker": ("1D76DB", "Docker/container work"),
    "data": ("C5DEF5", "Data pipeline work"),
    "evaluation": ("5319E7", "Evaluation and metrics"),
    "report": ("0E8A16", "Thesis report writing"),
    "rl": ("FBCA04", "Reinforcement learning"),
    "architecture": ("D4C5F9", "Architecture documentation and design"),
    "urgent": ("B60205", "Urgent priority"),
    "high": ("D93F0B", "High priority"),
    "medium": ("FBCA04", "Medium priority"),
    "low": ("C2E0C6", "Low priority"),
}


def run(cmd, allow_fail=False):
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    if result.returncode != 0 and not allow_fail:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def gh_json(cmd):
    out = run(cmd)
    return json.loads(out) if out else {}


def body(goal, description, acceptance, ai_instruction, notes=None):
    lines = [
        "## Goal", "", goal.strip(), "",
        "## Description", "", description.strip(), "",
        "## Acceptance Criteria", "",
    ]
    lines += [f"- {a}" for a in acceptance]
    lines += ["", "## Initial AI Agent Instruction", "", ai_instruction.strip()]
    if notes:
        lines += ["", "## Notes", "", notes.strip()]
    return "\n".join(lines).strip() + "\n"


TASKS = [
    {
        "title": "Define external repository dependency strategy",
        "milestone": "M1 — PoC Foundation",
        "labels": ["external", "dependency", "documentation", "architecture", "urgent"],
        "category": CATEGORY_ARCH,
        "priority": PRIORITY_URGENT,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_POC,
        "deadline": "2026-05-10",
        "body": body(
            "Define how external repositories are referenced, pinned and documented in the PoC.",
            "The project references FinRL, Gymnasium, ObjectRL, SDU_DataScienceTool and Zero Sum Public. The PoC should not blindly vendor large external repositories into the main source tree. Instead, create a lightweight external dependency strategy that documents URLs, roles, intended use and pinning approach. The first version can use a lock/manifest file with repo URL and commit placeholders.",
            [
                "external/ folder is defined in the repository structure",
                "external/README.md purpose is described",
                "external/external-repos.lock or equivalent manifest is defined",
                "FinRL, Gymnasium, ObjectRL, SDU_DataScienceTool and Zero Sum Public are listed with URL, role and pin field",
                "README.md is consistent with this strategy",
                "No external repository is cloned unless explicitly requested",
            ],
            "Create the external dependency manifest and README skeleton. Keep it simple and reproducible. Do not introduce submodules unless explicitly requested. The output should help future Docker builds or setup scripts know which external repositories may be fetched.",
            "This is a documentation and architecture task, not a heavy implementation task.",
        ),
    },
    {
        "title": "Define FinRL external dependency handling",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["external", "dependency", "finrl", "rl", "architecture", "high"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-11",
        "body": body(
            "Define how the PoC depends on FinRL without making the repository fragile.",
            "FinRL is central for the thesis and should be treated as an external framework. The project needs to document whether FinRL is installed as a package, cloned during Docker build, referenced through notebooks, or wrapped through dss-finrl-adapter.",
            [
                "FinRL URL is recorded in the external manifest",
                "installation strategy is documented",
                "expected use in research/notebooks/01_finrl_baseline.ipynb is described",
                "expected use in system/packages/dss-finrl-adapter is described",
                "known compatibility risks are listed",
                "task does not require completing the full FinRL experiment",
            ],
            "Create or update the FinRL dependency note. Focus on how the project will call/use FinRL, not on solving all FinRL training problems yet.",
        ),
    },
    {
        "title": "Define Gymnasium external dependency handling",
        "milestone": "M3 — Decision Support",
        "labels": ["external", "dependency", "gymnasium", "rl", "architecture", "high"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": body(
            "Define how Gymnasium is used as the custom trading environment interface.",
            "The thesis architecture includes a custom long-only stock investment environment. Gymnasium should define the reset/step interface, observations, actions and rewards. This task ensures the dependency is documented and connected to dss-env and research/notebooks/02_gymnasium_env.ipynb.",
            [
                "Gymnasium URL is recorded in the external manifest",
                "role as environment interface is documented",
                "connection to dss-env package is described",
                "connection to notebook 02_gymnasium_env is described",
                "initial environment assumptions are listed",
            ],
            "Create or update documentation that explains the intended Gymnasium role. Do not implement the full environment in this task.",
        ),
    },
    {
        "title": "Define ObjectRL external dependency handling",
        "milestone": "M3 — Decision Support",
        "labels": ["external", "dependency", "objectrl", "rl", "research", "medium"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": body(
            "Define whether ObjectRL is used as a reference/prototyping dependency.",
            "ObjectRL may be useful as a reference or prototyping library, but it should not block V1.0. This task documents its role and whether it belongs in external dependencies.",
            [
                "ObjectRL URL is recorded in the external manifest",
                "role is clearly marked as optional/reference unless actively used",
                "no blocking production dependency is introduced",
                "README or external dependency note explains when ObjectRL is relevant",
            ],
            "Document ObjectRL conservatively. Treat it as optional unless there is a clear implementation reason to use it in the V1.0 PoC.",
        ),
    },
    {
        "title": "Define SDU_DataScienceTool integration strategy",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["external", "dependency", "data", "api", "research", "high"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-11",
        "body": body(
            "Define how SDU_DataScienceTool can support data collection and API workflows.",
            "SDU_DataScienceTool may become a custom ingestion/data science helper. The project needs a clear plan for whether it is imported, called as a separate tool, copied into packages, or only referenced as an external dependency.",
            [
                "SDU_DataScienceTool URL is recorded in the external manifest",
                "intended role in data/API pipeline is described",
                "integration boundary is described",
                "no hidden coupling is introduced",
                "PoC can still run without it if not implemented yet",
            ],
            "Create the integration note and keep the boundary clean. Prefer a wrapper or adapter over copying large amounts of code.",
        ),
    },
    {
        "title": "Define Zero Sum Public frontend reference strategy",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["external", "dependency", "zero-sum", "frontend", "documentation", "medium"],
        "category": CATEGORY_DEV,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_POC,
        "deadline": "2026-05-13",
        "body": body(
            "Define how Zero Sum Public is used as a frontend/charting/design reference.",
            "Zero Sum Public may provide inspiration for portfolio/stock/decision views. It should be referenced as inspiration or technical reference without making the PoC dependent on its repository unless explicitly needed.",
            [
                "Zero Sum Public URL is recorded in the external manifest",
                "role as frontend/charting reference is documented",
                "copyright/licensing assumptions are not overstated",
                "no code is copied without explicit review",
                "frontend PoC remains independently runnable",
            ],
            "Document the reference role. Do not clone or copy code unless explicitly requested later.",
        ),
    },
    {
        "title": "Create research notebook index",
        "milestone": "M1 — PoC Foundation",
        "labels": ["research", "notebook", "documentation", "urgent"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_URGENT,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-10",
        "body": body(
            "Create a clear index for all research notebooks used in the thesis PoC.",
            "The project must run in two tracks: application implementation and academic research. The notebook index should explain the purpose of each notebook, expected inputs, expected outputs and how the results connect to the thesis report.",
            [
                "research/notebooks/README.md exists",
                "notebooks 00 to 06 are listed",
                "each notebook has purpose, inputs and outputs",
                "output folders for tables/figures are documented",
                "README explains which notebooks are mandatory for V1.0",
            ],
            "Create the notebook index as documentation first. Do not fill all notebooks with full experiments yet. Make it easy for an AI coding agent to open one notebook task and understand its role.",
        ),
    },
    {
        "title": "Create notebook 00 data check skeleton",
        "milestone": "M1 — PoC Foundation",
        "labels": ["research", "notebook", "data", "duckdb", "urgent"],
        "category": CATEGORY_DATA,
        "priority": PRIORITY_URGENT,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-10",
        "body": body(
            "Create the first data sanity-check notebook skeleton.",
            "Notebook 00 should verify that local paths, DuckDB connection, raw data folders and basic market data availability work. It should be the quickest way to confirm the research environment is alive.",
            [
                "research/notebooks/00_data_check.ipynb exists",
                "notebook loads environment/config paths",
                "notebook checks DuckDB connection",
                "notebook checks raw/parquet/csv folder availability",
                "notebook includes a small data availability table",
                "notebook can run without requiring full RL training",
            ],
            "Create a minimal notebook skeleton with markdown sections and lightweight cells. Prioritize reproducibility and sanity checks over advanced analysis.",
        ),
    },
    {
        "title": "Create notebook 01 FinRL baseline skeleton",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["research", "notebook", "finrl", "rl", "high"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-11",
        "body": body(
            "Create the FinRL baseline notebook skeleton.",
            "Notebook 01 should be the reproducible research entry point for the first FinRL baseline. It should document setup, selected assets, time period, data source, environment assumptions, training placeholder and evaluation outputs.",
            [
                "research/notebooks/01_finrl_baseline.ipynb exists",
                "notebook has sections for setup, data, environment, training, evaluation and export",
                "notebook documents V1.0 assumptions",
                "notebook writes or plans outputs to research/results",
                "notebook does not block the local DSS app if FinRL is not fully working yet",
            ],
            "Create a skeleton notebook with clear markdown and TODO cells. Make it compatible with the rest of the README architecture.",
        ),
    },
    {
        "title": "Create notebook 02 Gymnasium environment skeleton",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "notebook", "gymnasium", "rl", "high"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": body(
            "Create the custom Gymnasium environment notebook skeleton.",
            "Notebook 02 should demonstrate or outline the custom long-only stock investment environment. It should connect state, action and reward definitions to the dss-env package and later random-policy validation.",
            [
                "research/notebooks/02_gymnasium_env.ipynb exists",
                "state/action/reward sections are present",
                "long-only assumption is documented",
                "random policy validation section is present",
                "links to dss-env implementation path are included",
            ],
            "Create the notebook skeleton and keep implementation minimal. The main goal is to make the environment design reproducible and easy to iterate on.",
        ),
    },
    {
        "title": "Create notebook 03 baseline comparison skeleton",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "notebook", "evaluation", "risk", "high"],
        "category": CATEGORY_EVAL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": body(
            "Create the baseline comparison notebook skeleton.",
            "Notebook 03 should compare buy-and-hold, equal-weight and risk-adjusted decision-support output. It should create thesis-ready tables and figures where possible.",
            [
                "research/notebooks/03_baseline_comparison.ipynb exists",
                "buy-and-hold baseline section exists",
                "equal-weight baseline section exists",
                "risk-adjusted output section exists",
                "metrics include return, volatility, Sharpe-style metric and drawdown where possible",
                "outputs are written to research/results/tables and research/results/figures",
            ],
            "Create the notebook skeleton and connect it to the baseline comparison issue. Prefer simple transparent calculations before complex RL outputs.",
        ),
    },
    {
        "title": "Create notebook 04 IQN experiment skeleton",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "notebook", "iqn", "rl", "risk", "medium"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-15",
        "body": body(
            "Create the IQN-style experiment notebook skeleton.",
            "Notebook 04 should document how IQN/distributional RL could be evaluated or prototyped. For V1.0 this can remain a structured extension if full implementation is too large.",
            [
                "research/notebooks/04_iqn_experiment.ipynb exists",
                "IQN motivation section exists",
                "quantile output concept is documented",
                "connection to risk metrics is documented",
                "clearly marked as extension if not implemented for V1.0",
            ],
            "Create a skeleton notebook that supports thesis writing without requiring full IQN implementation immediately.",
        ),
    },
    {
        "title": "Create notebook 05 uncertainty proxy skeleton",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "notebook", "uncertainty", "risk", "medium"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-15",
        "body": body(
            "Create the uncertainty proxy notebook skeleton.",
            "Notebook 05 should document or prototype a simple uncertainty proxy, and optionally explain evidential uncertainty as a later extension. It should support transparent risk-aware decision output.",
            [
                "research/notebooks/05_uncertainty_proxy.ipynb exists",
                "uncertainty concept is documented",
                "simple proxy approach is described",
                "connection to decision cards and risk output is described",
                "limitations are explicitly documented",
            ],
            "Create a practical skeleton first. Do not overbuild evidential deep learning unless the core PoC is already working.",
        ),
    },
    {
        "title": "Create notebook 06 thesis figures skeleton",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["research", "notebook", "report", "evaluation", "high"],
        "category": CATEGORY_REPORT,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_REPORT,
        "deadline": "2026-05-14",
        "body": body(
            "Create the notebook that exports thesis-ready figures.",
            "Notebook 06 should gather results from DuckDB, CSV/Parquet or research/results and export figures/tables for the LaTeX report. It should be the final bridge from PoC outputs to the thesis.",
            [
                "research/notebooks/06_thesis_figures.ipynb exists",
                "input/output paths are documented",
                "exports go to research/results/figures and/or research/report/figures",
                "placeholder sections for architecture, data, baseline and decision audit figures exist",
                "notebook is designed to be re-run as results change",
            ],
            "Create a skeleton notebook for figure export. Keep plot generation deterministic and save outputs using stable filenames.",
        ),
    },
    {
        "title": "Create reproducible research experiment runner",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "experiment", "reproducibility", "devops", "high"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": body(
            "Create a reproducible experiment runner structure under research/experiments.",
            "The research track needs a way to run structured experiments outside notebooks. The first version can be lightweight: configs, runner script, output folder convention and metadata logging.",
            [
                "research/experiments/README.md exists",
                "experiment runner concept is documented",
                "config-driven execution is described",
                "outputs are written to research/results",
                "run ID or experiment ID convention is defined",
                "notebooks and system code can consume results later",
            ],
            "Create the structure and documentation for reproducible experiments. Keep it simple enough to implement quickly.",
        ),
    },
    {
        "title": "Create research experiment config templates",
        "milestone": "M3 — Decision Support",
        "labels": ["research", "experiment", "reproducibility", "documentation", "medium"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-12",
        "body": body(
            "Create initial YAML config templates for the planned research experiments.",
            "The README defines experiment configs for FinRL baseline, baseline comparison, risk-adjusted output and IQN extension. This task creates those config placeholders and documents required fields.",
            [
                "research/configs/experiment_001_finrl_baseline.yaml exists",
                "research/configs/experiment_002_baselines.yaml exists",
                "research/configs/experiment_003_risk_adjusted.yaml exists",
                "research/configs/experiment_004_iqn_extension.yaml exists",
                "common fields are documented",
                "configs are placeholders but syntactically valid YAML",
            ],
            "Create minimal config templates with comments. Do not hardcode secrets. Make the configs compatible with later experiment runner tasks.",
        ),
    },
    {
        "title": "Define research result artifact contract",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "evaluation", "reproducibility", "report", "high"],
        "category": CATEGORY_EVAL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_NEXT,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-13",
        "body": body(
            "Define the contract for research outputs that become thesis evidence.",
            "The project should standardize how experiment outputs are saved. This includes tables, figures, logs and metadata. The contract should make it easy to regenerate thesis evidence and connect results to audit/reproducibility claims.",
            [
                "research/results/README.md exists",
                "raw, processed, tables, figures and logs folders are described",
                "naming convention for artifacts is defined",
                "minimum metadata fields are defined",
                "connection to report figures/tables is described",
            ],
            "Create a clear artifact contract. Focus on reproducibility and easy thesis writing.",
        ),
    },
    {
        "title": "Define ML feature and indicator experiment plan",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "ml", "features", "risk", "medium"],
        "category": CATEGORY_RESEARCH,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-15",
        "body": body(
            "Define how ML-derived indicators may be explored without blocking V1.0.",
            "Some indicators may eventually come from machine learning models rather than simple technical indicators. This task creates a plan for which ML-derived indicators could be tested and how they would be compared to simpler baselines.",
            [
                "candidate ML-derived indicators are listed",
                "simple baseline indicators are listed",
                "evaluation method is described",
                "connection to feature pipeline is documented",
                "scope is clearly marked as optional/extension for V1.0",
            ],
            "Write a small experiment plan. Do not implement heavy ML models yet. Keep the core PoC focused.",
        ),
    },
    {
        "title": "Define hyperparameter tuning strategy for RL experiments",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["research", "rl", "hyperparameter", "model-selection", "medium"],
        "category": CATEGORY_RL,
        "priority": PRIORITY_MEDIUM,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-17",
        "body": body(
            "Define a lightweight hyperparameter tuning strategy for later RL experiments.",
            "Deep RL experiments can become expensive and noisy. This task defines a controlled approach to hyperparameter tuning so the thesis can discuss it without derailing the V1.0 PoC.",
            [
                "candidate hyperparameters are listed",
                "small-search strategy is described",
                "evaluation metrics are defined",
                "compute budget assumptions are documented",
                "reproducibility requirements are documented",
                "clearly marked as later/extension unless needed",
            ],
            "Create a concise tuning plan. Avoid overengineering. The plan should support future work and not block the current PoC.",
        ),
    },
    {
        "title": "Define model selection and comparison protocol",
        "milestone": "M4 — Risk, Uncertainty, Audit",
        "labels": ["research", "evaluation", "model-selection", "rl", "high"],
        "category": CATEGORY_EVAL,
        "priority": PRIORITY_HIGH,
        "roadmap": ROADMAP_LATER,
        "track": TRACK_RESEARCH,
        "deadline": "2026-05-14",
        "body": body(
            "Define how models and baselines are compared fairly.",
            "The thesis needs a clear comparison protocol across buy-and-hold, equal-weight, risk-adjusted decision support, FinRL baseline and later IQN/uncertainty variants.",
            [
                "models/baselines to compare are listed",
                "time periods and assets are defined or left as explicit parameters",
                "metrics are defined",
                "point-in-time constraints are included",
                "output table format is described",
                "limitations are documented",
            ],
            "Create a model comparison protocol that can be used in notebooks and report writing.",
        ),
    },
    {
        "title": "Document two-track workflow: DSS app and research experiments",
        "milestone": "M1 — PoC Foundation",
        "labels": ["documentation", "research", "architecture", "urgent"],
        "category": CATEGORY_ARCH,
        "priority": PRIORITY_URGENT,
        "roadmap": ROADMAP_NOW,
        "track": TRACK_POC,
        "deadline": "2026-05-10",
        "body": body(
            "Document the two-track workflow that keeps the project coherent.",
            "The project must be developed as both a working decision support system and a reproducible research/thesis environment. This task documents how system/ and research/ relate, what data they share, and how results move into the report.",
            [
                "two-track workflow is documented",
                "system/ responsibility is described",
                "research/ responsibility is described",
                "shared data/storage assumptions are described",
                "flow from research result to thesis figure/table is described",
                "flow from system output to audit/evidence is described",
            ],
            "Update the relevant README or docs file to make the two-track model explicit. This should help both human work and AI agent handoffs.",
        ),
    },
]


def create_label(name, color, description):
    existing = run([
        "gh", "label", "list", "--repo", GH_REPO, "--search", name,
        "--json", "name", "--jq", f'.[] | select(.name == "{name}") | .name'
    ], allow_fail=True)
    if existing.strip() == name:
        print(f"Label already exists: {name}")
        return
    run(["gh", "label", "create", name, "--repo", GH_REPO, "--color", color, "--description", description], allow_fail=True)
    print(f"Label ready: {name}")


def ensure_labels():
    print("Creating/finding labels...")
    for name, (color, description) in LABELS.items():
        create_label(name, color, description)
    print("")


def find_issue_number(title):
    issues = gh_json(["gh", "issue", "list", "--repo", GH_REPO, "--state", "all", "--limit", "300", "--json", "number,title"])
    for issue in issues:
        if issue.get("title") == title:
            return int(issue["number"])
    return None


def create_or_update_issue(task):
    title = task["title"]
    number = find_issue_number(title)
    labels = ",".join(task.get("labels", []))
    milestone = task.get("milestone")

    if number is None:
        cmd = ["gh", "issue", "create", "--repo", GH_REPO, "--title", title, "--body", task["body"]]
        if labels:
            cmd.extend(["--label", labels])
        if milestone:
            cmd.extend(["--milestone", milestone])
        url = run(cmd)
        number = int(url.rstrip("/").split("/")[-1])
        print(f"Created issue: #{number} — {title}")
    else:
        print(f"Issue already exists: #{number} — {title}")
        run(["gh", "issue", "edit", str(number), "--repo", GH_REPO, "--body", task["body"]])
        if labels:
            run(["gh", "issue", "edit", str(number), "--repo", GH_REPO, "--add-label", labels], allow_fail=True)
        if milestone:
            run(["gh", "issue", "edit", str(number), "--repo", GH_REPO, "--milestone", milestone], allow_fail=True)
    return number


def add_issue_to_project(issue_number):
    issue = gh_json(["gh", "issue", "view", str(issue_number), "--repo", GH_REPO, "--json", "url"])
    run(["gh", "project", "item-add", PROJECT_NUMBER, "--owner", OWNER, "--url", issue["url"]], allow_fail=True)


def get_project_id():
    project = gh_json(["gh", "project", "view", PROJECT_NUMBER, "--owner", OWNER, "--format", "json"])
    return project["id"]


def get_fields():
    data = gh_json(["gh", "project", "field-list", PROJECT_NUMBER, "--owner", OWNER, "--format", "json", "--limit", "100"])
    return {field["name"]: field for field in data.get("fields", [])}


def get_items():
    data = gh_json(["gh", "project", "item-list", PROJECT_NUMBER, "--owner", OWNER, "--format", "json", "--limit", "300"])
    out = {}
    for item in data.get("items", []):
        content = item.get("content") or {}
        title = content.get("title")
        if title:
            out[title] = item
    return out


def option_id(field, option_name):
    for option in field.get("options", []):
        if option.get("name") == option_name:
            return option.get("id")
    return None


def set_single(project_id, item_id, fields, field_name, option_name):
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return
    oid = option_id(field, option_name)
    if not oid:
        print(f"  WARNING: Option not found: {field_name} -> {option_name}")
        return
    run(["gh", "project", "item-edit", "--id", item_id, "--project-id", project_id, "--field-id", field["id"], "--single-select-option-id", oid])
    print(f"  Set {field_name} = {option_name}")


def set_date(project_id, item_id, fields, field_name, value):
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return
    run(["gh", "project", "item-edit", "--id", item_id, "--project-id", project_id, "--field-id", field["id"], "--date", value])
    print(f"  Set {field_name} = {value}")


def set_number(project_id, item_id, fields, field_name, value):
    field = fields.get(field_name)
    if not field:
        print(f"  WARNING: Field not found: {field_name}")
        return
    run(["gh", "project", "item-edit", "--id", item_id, "--project-id", project_id, "--field-id", field["id"], "--number", str(value)])
    print(f"  Set {field_name} = {value}")


def set_fields(task, project_id, fields, items):
    item = items.get(task["title"])
    if not item:
        print(f"  WARNING: Project item not found for: {task['title']}")
        return
    item_id = item["id"]
    print(f"Setting fields: {task['title']}")
    set_single(project_id, item_id, fields, "Status", STATUS_TODO)
    set_single(project_id, item_id, fields, "Category", task["category"])
    set_single(project_id, item_id, fields, "Priority", task["priority"])
    set_single(project_id, item_id, fields, "Roadmap", task["roadmap"])
    set_single(project_id, item_id, fields, "Track", task["track"])
    set_single(project_id, item_id, fields, "Percentage", PERCENT_0)
    set_date(project_id, item_id, fields, "Deadline", task["deadline"])
    set_number(project_id, item_id, fields, "Progress Number", 0)


ensure_labels()

print("Creating/finding issues and adding them to project...")
for task in TASKS:
    number = create_or_update_issue(task)
    add_issue_to_project(number)

print("\nFetching project metadata...")
project_id = get_project_id()
fields = get_fields()
items = get_items()

print("\nSetting project fields...")
for task in TASKS:
    set_fields(task, project_id, fields, items)

print("\nDone.\n")
print("Script 11 completed.")
print("Added/updated external dependency tasks + research notebook/experiment tasks.")
PY
