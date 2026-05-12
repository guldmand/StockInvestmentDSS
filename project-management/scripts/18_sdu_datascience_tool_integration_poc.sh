#!/usr/bin/env bash
set -euo pipefail

REPO="guldmand/StockInvestmentDSS"
PROJECT_OWNER="guldmand"
PROJECT_NUMBER="11"

echo "Using repo: ${REPO}"
echo "Using project: StockInvestmentDSS PoC Sprint (#${PROJECT_NUMBER})"
echo

echo "Checking GitHub auth..."
gh auth status
echo

echo "Creating/updating required labels..."

ensure_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  # --force makes label creation idempotent:
  # - creates the label if missing
  # - updates color/description if it already exists
  gh label create "$name" \
    --repo "$REPO" \
    --color "$color" \
    --description "$description" \
    --force >/dev/null

  echo "Label ready: $name"
}

ensure_label "sdu-datascience-tool" "0969DA" "SDU_DataScienceTool integration and adapter work."
ensure_label "api-ingestion" "1D76DB" "API ingestion, external data sources and adapters."
ensure_label "external-reference" "8B949E" "External project or repository used as implementation reference."
ensure_label "plotly" "5319E7" "Plotly/Dash visualization reference or implementation."
ensure_label "stock-news" "FBCA04" "Stock news and event data integration."
ensure_label "future-work" "6E7781" "Explicitly documented future work, not required for current PoC."
ensure_label "market-data" "0E8A16" "Market data acquisition, storage and data contracts."
ensure_label "documentation" "0075CA" "Documentation and project knowledge."
ensure_label "architecture" "5319E7" "Architecture, boundaries and system design."
ensure_label "application" "A2EEEF" "Application track work."
ensure_label "research" "0E8A16" "Research track work."
ensure_label "backend" "D4C5F9" "Backend implementation."
ensure_label "frontend" "C5DEF5" "Frontend implementation."
ensure_label "fast-layer" "FBCA04" "Online user-facing fast layer."
ensure_label "slow-layer" "BFDADC" "Offline/research/model training slow layer."
ensure_label "poc" "7057FF" "Proof-of-concept scope."

echo

echo "Checking GraphQL rate limit before Project updates..."
gh api graphql -f query='
query {
  rateLimit {
    remaining
    used
    resetAt
  }
}' --jq '.data.rateLimit | "  remaining: \(.remaining)\n  used:      \(.used)\n  resetAt:   \(.resetAt)"'
echo

PROJECT_ID="$(gh project view "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --format json --jq '.id')"

field_id() {
  local field_name="$1"
  gh project field-list "$PROJECT_NUMBER" \
    --owner "$PROJECT_OWNER" \
    --format json \
    --jq ".fields[] | select(.name == \"$field_name\") | .id" | head -n 1
}

option_id() {
  local field_name="$1"
  local option_name="$2"
  gh project field-list "$PROJECT_NUMBER" \
    --owner "$PROJECT_OWNER" \
    --format json \
    --jq ".fields[] | select(.name == \"$field_name\") | .options[]? | select(.name == \"$option_name\") | .id" | head -n 1
}

set_single_select_field() {
  local item_id="$1"
  local field_name="$2"
  local option_name="$3"

  local fid
  local oid
  fid="$(field_id "$field_name" || true)"
  oid="$(option_id "$field_name" "$option_name" || true)"

  if [[ -z "${fid}" || -z "${oid}" ]]; then
    echo "  Warning: could not set ${field_name}=${option_name} because field/option was not found."
    return 0
  fi

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$fid" \
    --single-select-option-id "$oid" >/dev/null

  echo "  ${field_name}: ${option_name}"
}

set_date_field() {
  local item_id="$1"
  local field_name="$2"
  local date_value="$3"

  local fid
  fid="$(field_id "$field_name" || true)"

  if [[ -z "${fid}" ]]; then
    echo "  Warning: could not set ${field_name}=${date_value} because field was not found."
    return 0
  fi

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$fid" \
    --date "$date_value" >/dev/null

  echo "  ${field_name}: ${date_value}"
}

set_number_field() {
  local item_id="$1"
  local field_name="$2"
  local number_value="$3"

  local fid
  fid="$(field_id "$field_name" || true)"

  if [[ -z "${fid}" ]]; then
    echo "  Warning: could not set ${field_name}=${number_value} because field was not found."
    return 0
  fi

  gh project item-edit \
    --id "$item_id" \
    --project-id "$PROJECT_ID" \
    --field-id "$fid" \
    --number "$number_value" >/dev/null

  echo "  ${field_name}: ${number_value}"
}

find_issue_number_by_title() {
  local title="$1"
  gh issue list \
    --repo "$REPO" \
    --state all \
    --search "$title in:title" \
    --json number,title \
    --jq ".[] | select(.title == \"$title\") | .number" | head -n 1
}

find_issue_url_by_number() {
  local number="$1"
  gh issue view "$number" --repo "$REPO" --json url --jq '.url'
}

find_project_item_id_by_issue_number() {
  local number="$1"
  gh project item-list "$PROJECT_NUMBER" \
    --owner "$PROJECT_OWNER" \
    --limit 200 \
    --format json \
    --jq ".items[] | select(.content.number == ${number}) | .id" | head -n 1
}

create_or_find_issue() {
  local title="$1"
  local labels="$2"
  local body="$3"

  local number
  number="$(find_issue_number_by_title "$title" || true)"

  if [[ -n "$number" ]]; then
    echo "Found existing issue: #${number} ${title}"
    find_issue_url_by_number "$number"
    return 0
  fi

  local body_file
  body_file="$(mktemp)"
  printf "%s\n" "$body" > "$body_file"

  echo "Creating issue: ${title}" >&2
  local url
  url="$(gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --body-file "$body_file" \
    --label "$labels")"

  rm -f "$body_file"
  echo "$url"
}

add_issue_to_project_and_update_fields() {
  local url="$1"
  local title="$2"
  local category="$3"
  local priority="$4"
  local roadmap="$5"
  local track="$6"
  local percentage="$7"
  local deadline="$8"
  local progress="$9"

  local number
  number="${url##*/}"

  echo "  Added to project."

  local item_id
  item_id="$(gh project item-add "$PROJECT_NUMBER" \
    --owner "$PROJECT_OWNER" \
    --url "$url" \
    --format json \
    --jq '.id' 2>/dev/null || true)"

  if [[ -z "$item_id" ]]; then
    sleep 2
    item_id="$(find_project_item_id_by_issue_number "$number" || true)"
  fi

  if [[ -z "$item_id" ]]; then
    echo "  Warning: could not find project item after adding issue."
    return 0
  fi

  echo "  Updating project fields..."
  set_single_select_field "$item_id" "Status" "Todo"
  set_single_select_field "$item_id" "Category" "$category"
  set_single_select_field "$item_id" "Priority" "$priority"
  set_single_select_field "$item_id" "Roadmap" "$roadmap"
  set_single_select_field "$item_id" "Track" "$track"
  set_single_select_field "$item_id" "Percentage" "$percentage"
  set_date_field "$item_id" "Deadline" "$deadline"
  set_number_field "$item_id" "Progress Number" "$progress"
}

ISSUE_171_BODY="$(cat <<'EOF'
Goal

Define how SDU_DataScienceTool should be used inside the StockInvestmentDSS PoC without turning it into the application framework.

Description

SDU_DataScienceTool is an existing external repository that can support API calls, data-source adapters, caching and reusable data-ingestion logic. The goal of this task is to define the adapter boundary before implementation, so the PoC uses existing working code where it helps, without coupling the entire DSS application to course-project code.

The intended role is:

StockInvestmentDSS backend
-> data adapter layer
-> SDU_DataScienceTool where useful
-> external APIs / market data / news data
-> DuckDB / runtime-data storage

Track

Application track: yes, because the backend application will consume the adapter.
Research track: yes, because market/news data may become part of the state representation or experiment inputs.
Shared track: yes, because the adapter belongs between ingestion, storage and application use.

Layer

Fast layer: yes, for online market/news lookups needed by the DSS UI.
Slow layer: yes, indirectly, if stored data is reused for notebooks, model features or evaluations.

System Context

External repos: SDU_DataScienceTool should be treated as an external reference/dependency, not blindly copied into the app.
Data pipeline: data must flow through backend APIs and storage boundaries, not directly from frontend to external APIs.
Storage / guldNAS / DuckDB: persisted data should land in DuckDB or runtime-data according to the existing PoC data rules.
Devices / infrastructure: must work locally first through Docker Compose.
Containers / Docker / k3s: implementation must remain container-friendly.
Research / Application split: adapter code must not mix dashboard UI logic with research notebook logic.

Implementation

Document:

- Which parts of SDU_DataScienceTool are relevant for the PoC.
- Which API/data-source features can be reused.
- Which parts should remain out of scope.
- How the adapter should be called from the StockInvestmentDSS backend.
- How retrieved data should be stored in DuckDB.
- How the adapter relates to FinRL/yfinance requirements.

Test

Verify that the document answers:

- What SDU_DataScienceTool is used for.
- What it is not used for.
- How it connects to backend, DuckDB and future research notebooks.
- How it avoids frontend-to-external-API coupling.

Acceptance Criteria

- Adapter boundary is documented.
- Relevant SDU_DataScienceTool features are listed.
- Out-of-scope parts are listed.
- Data flow is clear.
- The decision supports #166 and #172.
- No implementation is started before the boundary is clear.

Notes

This is a fast architecture task. Do not over-document. The goal is to prevent duplicated work and make the next implementation task safe.
EOF
)"

ISSUE_172_BODY="$(cat <<'EOF'
Goal

Integrate SDU_DataScienceTool as a small backend-side ingestion adapter for market/news API calls where it helps the PoC.

Description

The PoC should reuse existing working API/data-ingestion logic where possible instead of rebuilding every external data call from scratch. SDU_DataScienceTool may be used to support market data, news/event data, caching and API wrapper patterns.

This task should create a minimal integration path only. The goal is not to move the entire StockInvestmentDSS backend into SDU_DataScienceTool. The goal is to make useful API ingestion available through the backend and persist relevant outputs in DuckDB/runtime-data.

Track

Application track: yes, because backend APIs may expose stored/processed results to the frontend.
Research track: yes, because ingested data can later be used as research/model features.
Shared track: yes, because this touches API calls, storage and backend runtime behavior.

Layer

Fast layer: yes, for quick lookup endpoints used by the dashboard.
Slow layer: yes, later, when scheduled ingestion or feature generation is added.

System Context

External repos: SDU_DataScienceTool is an external reference/dependency.
Data pipeline: frontend must communicate with the StockInvestmentDSS backend, not directly with SDU_DataScienceTool.
Storage / guldNAS / DuckDB: data should be stored in local DuckDB/runtime-data first.
Devices / infrastructure: must run locally through Docker Compose.
Containers / Docker / k3s: adapter must work in containerized runtime.
Research / Application split: ingestion should be reusable by notebooks later, but not depend on notebooks.

Implementation

Implement minimal backend adapter structure, for example:

system/backend/app/services/
- data_sources/
- market_data_adapter.py
- news_data_adapter.py

Create only the minimum needed to support the next PoC step.

Potential first targets:

- fetch basic OHLCV data for a ticker
- fetch basic stock/news items for a ticker
- normalize response shape
- store or prepare for DuckDB persistence
- expose backend endpoint only if needed by current UI

Test

Verify:

- Adapter can be imported by backend.
- Backend still starts through Docker Compose.
- No frontend direct external API calls are introduced.
- A simple ticker lookup can return normalized data or a clearly documented placeholder.
- Errors are handled without crashing the app.

Acceptance Criteria

- Minimal SDU_DataScienceTool adapter exists or is explicitly documented as deferred after #171.
- Backend starts successfully.
- API ingestion boundary is respected.
- Data shape is documented or visible in code.
- Implementation supports future stock/news visualization work.
- No heavy RL training starts.

Notes

Keep this minimal. This task is about enabling data ingestion, not building the full dashboard.
EOF
)"

ISSUE_173_BODY="$(cat <<'EOF'
Goal

Map the DS808 clean_dashboard stock-news visualization into the StockInvestmentDSS stock detail view plan.

Description

The DS808 Visualization project contains an existing stock/news visualization idea using Dash/Plotly, especially the clean_dashboard version. The goal is to inspect that implementation and decide which pieces should be reused, translated or simplified for the StockInvestmentDSS PoC.

The PoC should not blindly copy the whole DS808 project. Instead, it should use the useful ideas:

- stock price chart
- news overlay or news markers
- stock-related news list
- ticker search / lookup pattern
- Plotly charting patterns
- API usage pattern through SDU_DataScienceTool

Track

Application track: yes, because this informs the stock detail frontend.
Research track: indirect, because news/price context may support model explanation or feature design.
Shared track: yes, because this touches data, backend contracts and frontend visualization.

Layer

Fast layer: yes, because the stock detail view is user-facing.
Slow layer: no, unless later reused for experiment plots.

System Context

External repos: DS808_Visualization is a private uploaded/reference project.
Data pipeline: visualizations must use backend endpoints or stored data, not direct frontend scraping.
Storage / guldNAS / DuckDB: chart/news data should align with DuckDB-backed market data.
Devices / infrastructure: must work locally first.
Containers / Docker / k3s: must fit the current Docker Compose PoC.
Research / Application split: visualization belongs to application track, not notebook-only output.

Implementation

Inspect DS808 clean_dashboard and document:

- Which files/components are relevant.
- Which API/data logic can inform SDU_DataScienceTool integration.
- Which Plotly chart ideas should be reused.
- Which parts are out of scope.
- What the minimal StockInvestmentDSS stock detail page should include.

Proposed minimal target:

/stocks or stock detail section:
- ticker input
- basic price chart placeholder or Plotly chart
- stock metadata
- latest news list
- later: news markers on price chart

Test

Verify:

- Mapping document exists.
- Minimal target UI is defined.
- Reuse boundary is clear.
- No large copy/paste migration is started before the mapping is approved.

Acceptance Criteria

- DS808 clean_dashboard is mapped to StockInvestmentDSS needs.
- Useful components/ideas are listed.
- Out-of-scope parts are listed.
- Stock detail MVP is defined.
- The result supports #167 and #172.

Notes

This task is not the implementation itself. It is the translation plan from prior working code to the current PoC stack.
EOF
)"

ISSUE_174_BODY="$(cat <<'EOF'
Goal

Document AI509_NLP_Agent as a future work layer, not as current PoC implementation.

Description

The AI509_NLP_Agent project is relevant as future work for an LLM/NLP agent layer on top of the StockInvestmentDSS. It should not distract from the current thesis implementation focus, which is RL, algorithmic trading, risk-aware decision support, FinRL and point-in-time data handling.

This task documents how the NLP agent could later extend the DSS after the core RL and application flow exists.

Track

Application track: indirect, because a future agent could become a UI/interaction layer.
Research track: yes, as future work around unstructured text and explanation.
Shared track: yes, because it may later connect to data ingestion, news, reports and model explanations.

Layer

Fast layer: later, if used for user-facing explanations or chat.
Slow layer: later, if used for summarization, embeddings or document processing.

System Context

External repos: AI509_NLP_Agent is a private uploaded/reference project.
Data pipeline: do not implement now.
Storage / guldNAS / DuckDB: future agent may read stored news/reports/model outputs.
Devices / infrastructure: not relevant for immediate implementation.
Containers / Docker / k3s: future container/service candidate.
Research / Application split: future explainability/assistant layer, not core PoC.

Implementation

Document:

- What AI509_NLP_Agent does.
- Why it is relevant later.
- Why it is out of scope now.
- How it could connect after the RL/DSS core works.
- Which thesis section may mention it as future work.

Potential future use cases:

- explain portfolio recommendations
- summarize news around selected tickers
- answer questions about market state
- summarize audit trail / decision evidence
- retrieve relevant reports or announcements

Test

Verify:

- It is clearly documented as future work.
- It does not add current implementation tasks.
- It does not shift focus away from FinRL/RL/algorithmic trading.

Acceptance Criteria

- Future work note exists.
- Current scope remains protected.
- AI/NLP agent relation to DSS is clearly explained.
- No implementation is started under this issue.

Notes

This is a scope-control task. The purpose is to acknowledge the prior work without letting it consume the current PoC timeline.
EOF
)"

echo "Creating/finding issues and updating project fields..."

declare -a CREATED_URLS=()

url="$(create_or_find_issue "Define SDU_DataScienceTool adapter strategy" "sdu-datascience-tool,api-ingestion,external-reference,architecture,documentation,poc" "$ISSUE_171_BODY")"
CREATED_URLS+=("$url")
add_issue_to_project_and_update_fields "$url" "Define SDU_DataScienceTool adapter strategy" "🏗️ Architecture" "⛰️ High" "✅ Now" "PoC" "□□□□□□□□□□ 0%" "2026-05-12" "0"
echo

url="$(create_or_find_issue "Integrate SDU_DataScienceTool for market and news API ingestion" "sdu-datascience-tool,api-ingestion,market-data,backend,application,research,poc" "$ISSUE_172_BODY")"
CREATED_URLS+=("$url")
add_issue_to_project_and_update_fields "$url" "Integrate SDU_DataScienceTool for market and news API ingestion" "📊 Data" "🗼 Urgent" "✅ Now" "PoC" "□□□□□□□□□□ 0%" "2026-05-13" "0"
echo

url="$(create_or_find_issue "Map DS808 clean_dashboard stock-news visualization to stock detail view" "external-reference,plotly,stock-news,frontend,application,documentation,poc" "$ISSUE_173_BODY")"
CREATED_URLS+=("$url")
add_issue_to_project_and_update_fields "$url" "Map DS808 clean_dashboard stock-news visualization to stock detail view" "⚙️ Development" "⛰️ High" "🔜 Next" "PoC" "□□□□□□□□□□ 0%" "2026-05-14" "0"
echo

url="$(create_or_find_issue "Document AI509 NLP Agent as future work layer" "external-reference,future-work,research,documentation,poc" "$ISSUE_174_BODY")"
CREATED_URLS+=("$url")
add_issue_to_project_and_update_fields "$url" "Document AI509 NLP Agent as future work layer" "🧾 Documentation" "🫣 Medium" "🔜 Next" "PoC" "□□□□□□□□□□ 0%" "2026-05-16" "0"
echo

echo "Done. Issues created/found:"
for url in "${CREATED_URLS[@]}"; do
  number="${url##*/}"
  title="$(gh issue view "$number" --repo "$REPO" --json title --jq '.title')"
  echo "- ${url} ${title}"
done

echo
echo "Script 18 done."
echo
echo "Recommended next issue:"
echo "  Define SDU_DataScienceTool adapter strategy"
