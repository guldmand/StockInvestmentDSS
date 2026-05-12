document.addEventListener("DOMContentLoaded", () => {
  // Redirect to dashboard if already authenticated
  if (StockInvestmentDSSAuth.getToken()) {
    window.location.replace("/dashboard");
    return;
  }

  const showLoginButton = document.querySelector("#show-login");
  const showRegisterButton = document.querySelector("#show-register");

  const loginForm = document.querySelector("#login-form");
  const registerForm = document.querySelector("#register-form");

  const loginFeedback = document.querySelector("#login-feedback");
  const registerFeedback = document.querySelector("#register-feedback");

  function showLogin() {
    loginForm.hidden = false;
    registerForm.hidden = true;

    showLoginButton.classList.add("auth-panel__tab--active");
    showRegisterButton.classList.remove("auth-panel__tab--active");

    loginFeedback.textContent = "";
    registerFeedback.textContent = "";
  }

  function showRegister() {
    loginForm.hidden = true;
    registerForm.hidden = false;

    showLoginButton.classList.remove("auth-panel__tab--active");
    showRegisterButton.classList.add("auth-panel__tab--active");

    loginFeedback.textContent = "";
    registerFeedback.textContent = "";
  }

  if (showLoginButton) {
    showLoginButton.addEventListener("click", showLogin);
  }

  if (showRegisterButton) {
    showRegisterButton.addEventListener("click", showRegister);
  }

  if (loginForm) {
    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();

      const username = document.querySelector("#login-username").value.trim();
      const password = document.querySelector("#login-password").value;

      loginFeedback.textContent = "Logging in...";

      try {
        await StockInvestmentDSSAuth.login(username, password);
        loginFeedback.textContent = "Login ok. Opening dashboard...";
        window.location.replace("/dashboard");
      } catch (error) {
        console.error(error);
        loginFeedback.textContent = error.message || "Login failed.";
      }
    });
  }

  if (registerForm) {
    registerForm.addEventListener("submit", async (event) => {
      event.preventDefault();

      const username = document.querySelector("#register-username").value.trim();
      const email = document.querySelector("#register-email").value.trim();
      const password = document.querySelector("#register-password").value;

      registerFeedback.textContent = "Creating account...";

      try {
        await StockInvestmentDSSAuth.register(username, email, password);
        registerFeedback.textContent = "Account created. You can now log in.";
        document.querySelector("#login-username").value = username;
        showLogin();
      } catch (error) {
        console.error(error);
        registerFeedback.textContent = error.message || "Account creation failed.";
      }
    });
  }

  showLogin();
});