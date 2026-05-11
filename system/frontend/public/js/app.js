const API_BASE_URL = "/api";

const backendStatus = document.querySelector("#backend-status");
const runtimeEnv = document.querySelector("#runtime-env");
const runtimeDuckdb = document.querySelector("#runtime-duckdb");
const runtimeGuldnas = document.querySelector("#runtime-guldnas");

async function loadBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();

    backendStatus.textContent = `Backend ${data.status}`;
    backendStatus.classList.remove("status-card__value--pending", "status-card__value--error", "status--error");
    backendStatus.classList.add("status-card__value--ok", "status--ok");
  } catch (error) {
    backendStatus.textContent = "Backend unavailable";
    backendStatus.classList.remove("status-card__value--pending", "status-card__value--ok", "status--ok");
    backendStatus.classList.add("status-card__value--error", "status--error");
    console.error(error);
  }
}

async function loadRuntimeConfig() {
  try {
    const response = await fetch(`${API_BASE_URL}/config/runtime`);

    if (!response.ok) {
      throw new Error(`Runtime config returned ${response.status}`);
    }

    const data = await response.json();

    runtimeEnv.textContent = data.app_env ?? "-";
    runtimeDuckdb.textContent = data.duckdb_path ?? "-";
    runtimeGuldnas.textContent = data.guldnas_duckdb_path ?? "-";
  } catch (error) {
    runtimeEnv.textContent = "unavailable";
    runtimeDuckdb.textContent = "unavailable";
    runtimeGuldnas.textContent = "unavailable";
    console.error(error);
  }
}

loadBackendStatus();
loadRuntimeConfig();