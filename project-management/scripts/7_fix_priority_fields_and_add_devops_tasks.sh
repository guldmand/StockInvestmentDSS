#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StockInvestmentDSS — Script 7
# Fix priority labels -> project Priority field
# Add missing CI/CD / DevOps best-practice tasks
#
# Does:
# 1) Reads existing issues
# 2) Infers priority from labels:
#      urgent / priority:urgent  -> 🗼 Urgent
#      high   / priority:high    -> ⛰️ High
#      medium / priority:medium  -> 🫣 Medium
#      low    / priority:low     -> 🌈 Low
# 3) Sets the GitHub Project "Priority" field
# 4) Removes priority labels from issues
# 5) Deletes priority label definitions from repo
# 6) Creates extra CI/CD tasks if missing
# 7) Adds those tasks to project and sets fields
# ============================================================

OWNER="guldmand"
REPO="StockInvestmentDSS"
GH_REPO="$OWNER/$REPO"
PROJECT_NUMBER="11"
PROJECT_TITLE="StockInvestmentDSS PoC Sprint"

# ------------------------------------------------------------
# Read generated project.env safely if it exists
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

# ------------------------------------------------------------
# Python detection
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
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

# ------------------------------------------------------------
# Run Python orchestration
# ------------------------------------------------------------

$PYTHON_BIN - <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

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

P0 = "□□□□□□□□□□ 0%"

NOW = "✅ Now"
NEXT = "🔜 Next"
LATER = "🗓️ Later"

PRIORITY_OPTIONS = {
    "urgent": "🗼 Urgent",
    "high": "⛰️ High",
    "medium": "🫣 Medium",
    "low": "🌈 Low",
}

PRIORITY_LABELS = {
    "priority:urgent": "urgent",
    "priority:high": "high",
    "priority:medium": "medium",
    "priority:low": "low",
    "urgent": "urgent",
    "high": "high",
    "medium": "medium",
    "low": "low",
}

PRIORITY_DELETE_LABELS = [
    "priority:urgent",
    "priority:high",
    "priority:medium",
    "priority:low",
    "urgent",
    "high",
    "medium",
    "low",
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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
        print("Wait until resetAt if the script fails.")
        print("The script is idempotent and can safely be rerun.")

def create_label(name, color, description):
    result = subprocess.run([
        "gh", "label", "create", name,
        "--repo", GH_REPO,
        "--color", color,
        "--description", description,
        "--force",
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print(f"Label ready: {name}")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        print(f"Warning: label could not be created/updated: {name}")
        print(msg)

def delete_repo_label_if_exists(label_name):
    result = subprocess.run([
        "gh", "label", "delete", label_name,
        "--repo", GH_REPO,
        "--yes",
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print(f"Deleted repo label: {label_name}")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        if "not found" in msg.lower() or "could not resolve" in msg.lower():
            print(f"Repo label not present: {label_name}")
        else:
            print(f"Warning: could not delete repo label: {label_name}")
            print(msg)

def remove_label_from_issue(issue_number, label_name):
    encoded = quote(label_name, safe="")
    result = subprocess.run([
        "gh", "api",
        "-X", "DELETE",
        f"repos/{GH_REPO}/issues/{issue_number}/labels/{encoded}",
    ], text=True, capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print(f"  Removed label: {label_name}")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        if "not found" in msg.lower() or "does not exist" in msg.lower():
            pass
        else:
            print(f"  Warning: could not remove label {label_name}")
            print(f"  {msg}")

def fetch_all_issues_rest():
    data = gh_json([
        "gh", "api",
        "--paginate",
        "--slurp",
        f"repos/{GH_REPO}/issues?state=all&per_page=100"
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
                "labels": [label["name"] for label in item.get("labels", [])],
            })

    return issues

def infer_priority_from_labels(label_names):
    normalized = set(label_names)

    # Priority order is intentional.
    for label in ["priority:urgent", "urgent"]:
        if label in normalized:
            return "urgent"

    for label in ["priority:high", "high"]:
        if label in normalized:
            return "high"

    for label in ["priority:medium", "medium"]:
        if label in normalized:
            return "medium"

    for label in ["priority:low", "low"]:
        if label in normalized:
            return "low"

    return None

def issue_find_exact(title):
    issues = fetch_all_issues_rest()
    for issue in issues:
        if issue["title"] == title:
            return issue
    return None

def create_or_get_issue(task):
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
        "--milestone", task["milestone"],
    ]

    url = run(cmd)

    created = issue_find_exact(task["title"])
    if created:
        return created

    return {
        "number": None,
        "title": task["title"],
        "url": url,
        "labels": task["labels"],
    }

def add_issue_to_project(issue_url):
    if not issue_url:
        return

    result = subprocess.run([
        "gh", "project", "item-add", PROJECT_NUMBER,
        "--owner", OWNER,
        "--url", issue_url
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

def field_id(fields, name):
    for field in fields:
        if field.get("name") == name:
            return field.get("id")
    return None

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

def update_date(project_id, item_id, field_id_value, date_value):
    query = """
    mutation(
      $projectId: ID!,
      $itemId: ID!,
      $fieldId: ID!,
      $date: Date!
    ) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { date: $date }
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
        "-f", f"date={date_value}",
    ])

def set_new_task_fields(project_id, item_id, fields, task):
    priority_field_id = field_id(fields, "Priority")
    roadmap_field_id = field_id(fields, "Roadmap")
    percentage_field_id = field_id(fields, "Percentage")
    deadline_field_id = field_id(fields, "Deadline")

    priority_option = option_id(fields, "Priority", PRIORITY_OPTIONS[task["priority"]])
    roadmap_option = option_id(fields, "Roadmap", task["roadmap"])
    percentage_option = option_id(fields, "Percentage", P0)

    if priority_field_id and priority_option:
        update_single_select(project_id, item_id, priority_field_id, priority_option)
        print(f"  Priority:   {PRIORITY_OPTIONS[task['priority']]}")

    if roadmap_field_id and roadmap_option:
        update_single_select(project_id, item_id, roadmap_field_id, roadmap_option)
        print(f"  Roadmap:    {task['roadmap']}")

    if percentage_field_id and percentage_option:
        update_single_select(project_id, item_id, percentage_field_id, percentage_option)
        print(f"  Percentage: {P0}")

    if deadline_field_id:
        update_date(project_id, item_id, deadline_field_id, task["deadline"])
        print(f"  Deadline:   {task['deadline']}")

# ------------------------------------------------------------
# Extra CI/CD / DevOps tasks
# ------------------------------------------------------------

extra_tasks = [
    {
        "title": "Define branch protection and required checks",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "ci", "documentation"],
        "priority": "medium",
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Define branch protection rules for the repository.

## Acceptance criteria
- main branch protection is documented
- required CI checks are listed
- pull request review expectations are defined
- direct commits to main are discouraged
- this is documented as a PoC-safe DevOps best practice

## Notes
This is not blocking V1.0, but it improves project hygiene."""
    },
    {
        "title": "Create build-and-push container image workflow",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["devops", "ci", "docker"],
        "priority": "high",
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Create a GitHub Actions workflow that builds and optionally pushes the application container image.

## Acceptance criteria
- workflow builds the app Dockerfile
- image tag strategy is documented
- push target is documented or left disabled for PoC
- workflow can later support test/prod deployment

## Suggested file
.github/workflows/build-and-push.yml"""
    },
    {
        "title": "Add CI smoke test for local app startup",
        "milestone": "M3 — Decision Support",
        "labels": ["devops", "ci", "test", "frontend", "backend"],
        "priority": "high",
        "roadmap": LATER,
        "deadline": "2026-05-12",
        "body": """## Goal
Add a lightweight smoke test that verifies the app can start.

## Acceptance criteria
- app imports without crashing
- app startup command is tested
- login route/front page route is checked if technically feasible
- CI fails if startup breaks

## Scope
Keep it simple. This should not become a full end-to-end test suite."""
    },
    {
        "title": "Add Docker Compose smoke test workflow",
        "milestone": "M5 — Demo + Report Figures",
        "labels": ["devops", "ci", "docker", "test"],
        "priority": "medium",
        "roadmap": LATER,
        "deadline": "2026-05-14",
        "body": """## Goal
Verify that docker compose can start the PoC stack.

## Acceptance criteria
- docker compose config validates
- containers can build
- app container can start
- health check or logs confirm startup
- workflow is allowed to remain optional if CI runtime becomes too slow

## Suggested file
.github/workflows/docker-compose-smoke.yml"""
    },
    {
        "title": "Add deployment health check and rollback gate",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "monitoring", "rollback", "test-env"],
        "priority": "medium",
        "roadmap": LATER,
        "deadline": "2026-05-17",
        "body": """## Goal
Define a simple deployment health check and rollback gate.

## Acceptance criteria
- health endpoint or app-start check is defined
- deployment failure criteria are documented
- rollback trigger is documented
- test environment deployment should not continue if health check fails

## Scope
This can be documentation-first for V1.0 and implementation later."""
    },
    {
        "title": "Document GitHub Actions secrets and environment variables",
        "milestone": "M2 — Strategy + Portfolio",
        "labels": ["devops", "ci", "environment", "documentation"],
        "priority": "high",
        "roadmap": NEXT,
        "deadline": "2026-05-11",
        "body": """## Goal
Document the secrets and environment variables needed by GitHub Actions.

## Acceptance criteria
- required secrets are listed
- local .env and CI secrets are separated
- no real secrets are committed
- test/prod environment differences are documented
- README4 or docs/devops.md references this setup"""
    },
    {
        "title": "Add release tagging and versioning workflow",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "ci", "documentation"],
        "priority": "medium",
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Define a simple versioning and release tagging approach.

## Acceptance criteria
- V1.0 tag strategy is documented
- release notes idea is documented
- demo build/version can be identified
- future model/data versions can reference release tags"""
    },
    {
        "title": "Add security and dependency scanning workflow",
        "milestone": "M6 — Buffer / v1.1",
        "labels": ["devops", "ci", "documentation"],
        "priority": "medium",
        "roadmap": LATER,
        "deadline": "2026-05-15",
        "body": """## Goal
Add or document basic dependency/security scanning.

## Acceptance criteria
- Dependabot is linked to CI thinking
- Python dependency scanning is considered
- GitHub Actions dependency scanning is considered
- any skipped security checks are documented as PoC limitations

## Notes
This is useful for a thesis system with a future production path."""
    },
]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

print("")
print("Checking GraphQL rate before running...")
check_graphql_rate_limit()

print("")
print("Creating required non-priority labels for extra CI/CD tasks...")

required_labels = [
    ("devops", "5319E7", "CI/CD, deployment workflows and automation"),
    ("ci", "5319E7", "GitHub Actions CI checks"),
    ("docker", "2496ED", "Dockerfiles, Docker Compose and container setup"),
    ("test", "D4C5F9", "Manual or automated tests"),
    ("frontend", "BFD4F2", "Frontend, UI, charts, dashboard"),
    ("backend", "D4C5F9", "Backend, API, app logic"),
    ("monitoring", "5319E7", "Monitoring and logging"),
    ("rollback", "5319E7", "Rollback strategy"),
    ("test-env", "C2E0C6", "Test environment"),
    ("environment", "C2E0C6", ".env files and environment configuration"),
    ("documentation", "0075CA", "README, docs, runbooks"),
]

for name, color, description in required_labels:
    create_label(name, color, description)

print("")
print("Fetching project metadata...")
project_id = get_project_id()
fields = get_project_fields()

priority_field_id = field_id(fields, "Priority")

if not priority_field_id:
    print("ERROR: Project field not found: Priority")
    sys.exit(1)

priority_option_ids = {}
for key, option_name in PRIORITY_OPTIONS.items():
    oid = option_id(fields, "Priority", option_name)
    if not oid:
        print(f"ERROR: Priority option not found: {option_name}")
        print("Check the exact spelling/emojis in the Priority field.")
        sys.exit(1)
    priority_option_ids[key] = oid

print("")
print("Fetching project items...")
items = get_project_items()

print("")
print("Creating/finding extra CI/CD issues...")

created_or_found_extra_titles = []

for task in extra_tasks:
    issue = create_or_get_issue(task)
    add_issue_to_project(issue.get("url"))
    created_or_found_extra_titles.append(task["title"])
    time.sleep(0.15)

time.sleep(2)

print("")
print("Refreshing project items after adding extra tasks...")
items = get_project_items()

print("")
print("Setting fields for extra CI/CD tasks...")

for task in extra_tasks:
    item_id = item_id_by_title(items, task["title"])
    if not item_id:
        print(f"WARNING: Project item not found for extra task: {task['title']}")
        continue

    print(f"Updating extra task fields: {task['title']}")
    set_new_task_fields(project_id, item_id, fields, task)
    time.sleep(0.15)

print("")
print("Fetching all issues from repository...")
issues = fetch_all_issues_rest()

print("")
print("Moving priority from labels to Project Priority field...")

updated_count = 0
skipped_count = 0

for issue in issues:
    priority_key = infer_priority_from_labels(issue["labels"])

    if not priority_key:
        skipped_count += 1
        continue

    item_id = item_id_by_title(items, issue["title"])

    if not item_id:
        print(f"WARNING: Project item not found for issue #{issue['number']}: {issue['title']}")
        continue

    option_id_value = priority_option_ids[priority_key]

    print(f"Updating priority field: #{issue['number']} {issue['title']}")
    print(f"  From label -> Priority field: {PRIORITY_OPTIONS[priority_key]}")

    update_single_select(
        project_id=project_id,
        item_id=item_id,
        field_id_value=priority_field_id,
        option_id_value=option_id_value,
    )

    updated_count += 1
    time.sleep(0.12)

print("")
print(f"Priority field updates completed: {updated_count}")
print(f"Issues without priority label skipped: {skipped_count}")

print("")
print("Removing priority labels from all issues...")

removed_from_issues = 0

for issue in issues:
    labels_to_remove = [label for label in issue["labels"] if label in PRIORITY_DELETE_LABELS]

    if not labels_to_remove:
        continue

    print(f"Cleaning issue labels: #{issue['number']} {issue['title']}")

    for label in labels_to_remove:
        remove_label_from_issue(issue["number"], label)
        removed_from_issues += 1
        time.sleep(0.05)

print("")
print(f"Priority labels removed from issues: {removed_from_issues}")

print("")
print("Deleting priority label definitions from repository...")

for label in PRIORITY_DELETE_LABELS:
    delete_repo_label_if_exists(label)

print("")
print("Final GraphQL rate check...")
check_graphql_rate_limit()

print("")
print("Done.")
print("")
print("Script 7 completed.")
print("")
print("What was done:")
print("- Project Priority field set from old priority labels")
print("- priority labels removed from issues")
print("- priority label definitions deleted from repo")
print("- extra CI/CD / DevOps best-practice tasks created if missing")
print("- extra CI/CD tasks added to project and configured with Priority/Roadmap/Percentage/Deadline")
PY