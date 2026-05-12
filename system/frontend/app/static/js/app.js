const API_BASE_URL = "/api";

const backendStatus = document.querySelector("#backend-status");
const runtimeEnv = document.querySelector("#runtime-env");
const runtimeDuckdb = document.querySelector("#runtime-duckdb");
const runtimeGuldnas = document.querySelector("#runtime-guldnas");
const currentUser = document.querySelector("#current-user");
const logoutButton = document.querySelector("#logout-button");

function setStatus(element, text, stateClass) {
  if (!element) return;

  element.textContent = text;
  element.classList.remove(
    "status-card__value--pending",
    "status-card__value--ok",
    "status-card__value--error"
  );
  element.classList.add(stateClass);
}

function getAuthToken() {
  return localStorage.getItem("stockinvestmentdss_auth_token");
}

function getAuthHeaders() {
  const token = getAuthToken();

  if (!token) {
    return {};
  }

  return {
    Authorization: `Bearer ${token}`,
  };
}

async function loadBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    setStatus(
      backendStatus,
      `Backend ${data.status ?? "ok"}`,
      "status-card__value--ok"
    );
  } catch (error) {
    console.error("Backend health check failed:", error);
    setStatus(
      backendStatus,
      "Backend unavailable",
      "status-card__value--error"
    );
  }
}

async function loadRuntimeConfig() {
  try {
    const response = await fetch(`${API_BASE_URL}/config/runtime`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Runtime config returned ${response.status}`);
    }

    const data = await response.json();

    if (runtimeEnv) runtimeEnv.textContent = data.app_env ?? "-";
    if (runtimeDuckdb) runtimeDuckdb.textContent = data.duckdb_path ?? "-";
    if (runtimeGuldnas) runtimeGuldnas.textContent = data.guldnas_duckdb_path ?? "-";
  } catch (error) {
    console.error("Runtime config check failed:", error);

    if (runtimeEnv) runtimeEnv.textContent = "unavailable";
    if (runtimeDuckdb) runtimeDuckdb.textContent = "unavailable";
    if (runtimeGuldnas) runtimeGuldnas.textContent = "unavailable";
  }
}

async function loadCurrentUser() {
  if (!currentUser) return;

  const token = getAuthToken();

  if (!token) {
    currentUser.textContent = "Not logged in";
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Session returned ${response.status}`);
    }

    const data = await response.json();
    currentUser.textContent = `Logged in as ${data.username ?? "user"}`;
  } catch (error) {
    console.error("Session check failed:", error);
    currentUser.textContent = "Session unavailable";
  }
}

async function logout() {
  try {
    await fetch(`${API_BASE_URL}/auth/logout`, {
      method: "POST",
      headers: getAuthHeaders(),
    });
  } catch (error) {
    console.error("Logout request failed:", error);
  } finally {
    localStorage.removeItem("stockinvestmentdss_auth_token");
    window.location.href = "/login";
  }
}

if (logoutButton) {
  logoutButton.addEventListener("click", logout);
}

loadCurrentUser();
loadBackendStatus();
loadRuntimeConfig();