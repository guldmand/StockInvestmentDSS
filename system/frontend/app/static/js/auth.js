const StockInvestmentDSSAuth = (() => {
  const TOKEN_KEY = "stockinvestmentdss_auth_token";
  const USER_KEY = "stockinvestmentdss_auth_user";
  const API_BASE_URL = "/api";

  function getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
  }

  function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }

  function setUser(user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  function getUser() {
    const rawUser = localStorage.getItem(USER_KEY);

    if (!rawUser) {
      return null;
    }

    try {
      return JSON.parse(rawUser);
    } catch {
      return null;
    }
  }

  function getAuthHeaders() {
    const token = getToken();

    const headers = {
      "Content-Type": "application/json",
    };

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    return headers;
  }

  async function apiFetch(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: "same-origin",
      headers: {
        ...getAuthHeaders(),
        ...(options.headers || {}),
      },
    });

    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    const data = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof data === "object" && data !== null
          ? data.detail || data.message || JSON.stringify(data)
          : data;

      throw new Error(message || `Request failed with status ${response.status}`);
    }

    return data;
  }

  async function login(username, password) {
    const data = await apiFetch("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username,
        password,
      }),
    });

    const token = data.access_token || data.token || data.session_token;

    if (!token) {
      throw new Error("Login succeeded, but no token was returned by the backend.");
    }

    setToken(token);

    if (data.user) {
      setUser(data.user);
    }

    return data;
  }

  async function register(username, email, password) {
    return apiFetch("/auth/register", {
      method: "POST",
      body: JSON.stringify({
        username,
        email: email || null,
        password,
      }),
    });
  }

  async function me() {
    return apiFetch("/auth/me", {
      method: "GET",
    });
  }

  async function logout() {
    try {
      await apiFetch("/auth/logout", {
        method: "POST",
      });
    } catch (error) {
      console.warn("Backend logout failed, clearing local session anyway:", error);
    } finally {
      clearToken();
    }
  }

  function requireLogin() {
    if (!getToken()) {
      window.location.replace("/login");
      return false;
    }

    return true;
  }

  return {
    API_BASE_URL,
    getToken,
    setToken,
    clearToken,
    setUser,
    getUser,
    getAuthHeaders,
    apiFetch,
    login,
    register,
    me,
    logout,
    requireLogin,
  };
})();