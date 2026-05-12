"""FastAPI frontend for the StockInvestmentDSS PoC."""

import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://backend:8000")
FRONTEND_SESSION_COOKIE = "stockinvestmentdss_frontend_session"
FRONTEND_SESSION_MAX_AGE_SECONDS = 60 * 60 * 12


app = FastAPI(
    title="StockInvestmentDSS Frontend",
    description="Jinja2 frontend shell for the StockInvestmentDSS PoC.",
    version="0.1.0",
    docs_url="/frontend-docs",
    redoc_url=None,
    openapi_url="/frontend-openapi.json",
)

templates = Jinja2Templates(directory="app/templates")

app.mount("/css", StaticFiles(directory="app/static/css"), name="css")
app.mount("/js", StaticFiles(directory="app/static/js"), name="js")
app.mount("/images", StaticFiles(directory="app/static/images"), name="images")


def get_frontend_session_token(request: Request) -> str | None:
    """Return the local PoC frontend session token if present."""
    return request.cookies.get(FRONTEND_SESSION_COOKIE)


def copy_request_headers(request: Request) -> dict[str, str]:
    """Copy safe request headers for backend proxy calls."""
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Route users to login or dashboard based on local frontend session cookie."""
    if get_frontend_session_token(request):
        return RedirectResponse(url="/dashboard", status_code=303)

    return RedirectResponse(url="/login", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Render the dashboard only when the local frontend session cookie exists."""
    if not get_frontend_session_token(request):
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page_title": "Dashboard | StockInvestmentDSS",
            "body_class": "page page--dashboard",
            "active_nav": "dashboard",
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login(request: Request):
    """Render login page, unless the user already has a local frontend session."""
    if get_frontend_session_token(request):
        return RedirectResponse(url="/dashboard", status_code=303)

    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "page_title": "Login | StockInvestmentDSS",
            "body_class": "page page--auth",
            "active_nav": "login",
        },
    )


@app.get("/frontend-health")
def frontend_health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
async def proxy_login(request: Request):
    """Proxy login to backend and set local frontend session cookie on success."""
    body = await request.body()

    async with httpx.AsyncClient() as client:
        backend_response = await client.post(
            f"{BACKEND_BASE_URL}/auth/login",
            content=body,
            headers=copy_request_headers(request),
            timeout=30.0,
        )

    response = Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        media_type=backend_response.headers.get("content-type", "application/json"),
    )

    if backend_response.status_code == 200:
        try:
            data = backend_response.json()
        except ValueError:
            data = {}

        token = (
            data.get("access_token") or data.get("token") or data.get("session_token")
        )

        if token:
            response.set_cookie(
                key=FRONTEND_SESSION_COOKIE,
                value=token,
                max_age=FRONTEND_SESSION_MAX_AGE_SECONDS,
                path="/",
                httponly=True,
                secure=False,
                samesite="lax",
            )

    return response


@app.post("/api/auth/logout")
async def proxy_logout(request: Request):
    """Proxy logout to backend and always clear local frontend session cookie."""
    token = get_frontend_session_token(request)
    headers = copy_request_headers(request)

    if token and "authorization" not in {key.lower() for key in headers}:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_BASE_URL}/auth/logout",
                headers=headers,
                timeout=30.0,
            )
    except httpx.HTTPError:
        pass

    response = Response(
        content='{"status":"ok"}',
        status_code=200,
        media_type="application/json",
    )
    response.delete_cookie(
        key=FRONTEND_SESSION_COOKIE,
        path="/",
    )

    return response


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_api(path: str, request: Request):
    """Proxy frontend /api/* calls to the backend container."""
    target_url = f"{BACKEND_BASE_URL}/{path}"

    headers = copy_request_headers(request)
    body = await request.body()

    async with httpx.AsyncClient() as client:
        backend_response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
            timeout=30.0,
        )

    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        headers={
            key: value
            for key, value in backend_response.headers.items()
            if key.lower()
            not in {"content-encoding", "transfer-encoding", "connection"}
        },
        media_type=backend_response.headers.get("content-type"),
    )


@app.get("/docs")
async def proxy_docs():
    """Expose backend Swagger UI through the frontend origin."""
    async with httpx.AsyncClient() as client:
        backend_response = await client.get(f"{BACKEND_BASE_URL}/docs")

    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        media_type=backend_response.headers.get("content-type", "text/html"),
    )


@app.get("/openapi.json")
async def proxy_openapi():
    """Expose backend OpenAPI JSON through the frontend origin."""
    async with httpx.AsyncClient() as client:
        backend_response = await client.get(f"{BACKEND_BASE_URL}/openapi.json")

    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        media_type="application/json",
    )
