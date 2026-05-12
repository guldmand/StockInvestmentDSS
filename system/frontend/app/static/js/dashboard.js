document.addEventListener("DOMContentLoaded", () => {
  if (!StockInvestmentDSSAuth.getToken()) {
    window.location.replace("/login");
    return;
  }

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

  async function loadBackendStatus() {
    try {
      const data = await StockInvestmentDSSAuth.apiFetch("/health", {
        method: "GET",
      });

      setStatus(
        backendStatus,
        `Backend ${data.status || "ok"}`,
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
      const data = await StockInvestmentDSSAuth.apiFetch("/config/runtime", {
        method: "GET",
      });

      if (runtimeEnv) runtimeEnv.textContent = data.app_env || "-";
      if (runtimeDuckdb) runtimeDuckdb.textContent = data.duckdb_path || "-";
      if (runtimeGuldnas) {
        runtimeGuldnas.textContent = data.guldnas_duckdb_path || "-";
      }
    } catch (error) {
      console.error("Runtime config check failed:", error);

      if (runtimeEnv) runtimeEnv.textContent = "unavailable";
      if (runtimeDuckdb) runtimeDuckdb.textContent = "unavailable";
      if (runtimeGuldnas) runtimeGuldnas.textContent = "unavailable";
    }
  }

  async function loadCurrentUser() {
    if (!currentUser) return;

    try {
      const data = await StockInvestmentDSSAuth.me();
      currentUser.textContent = `Logged in as ${
        data.username || data.email || "user"
      }`;
    } catch (error) {
      console.warn("Session check failed:", error);
      StockInvestmentDSSAuth.clearToken();
      window.location.replace("/login");
    }
  }

  async function handleLogout() {
    await StockInvestmentDSSAuth.logout();
    window.location.replace("/login");
  }

  if (logoutButton) {
    logoutButton.addEventListener("click", handleLogout);
  }

  loadCurrentUser();
  loadBackendStatus();
  loadRuntimeConfig();
});