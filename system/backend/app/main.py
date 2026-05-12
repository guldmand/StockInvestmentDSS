"""FastAPI entrypoint for the StockInvestmentDSS PoC backend."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets
import uuid
from typing import Optional

import bcrypt
import duckdb
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings

SESSION_TTL_HOURS = 24


app = FastAPI(
    title="StockInvestmentDSS API",
    description="PoC API for data-driven stock investment decision support.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------


def get_runtime_data_path() -> Path:
    runtime_path = Path(settings.runtime_data_path)
    runtime_path.mkdir(parents=True, exist_ok=True)
    return runtime_path


def get_duckdb_path() -> Path:
    duckdb_path = Path(settings.duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb_path


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(get_duckdb_path()))


def init_auth_tables() -> None:
    """Create minimal persistent PoC auth tables in DuckDB."""
    get_runtime_data_path()

    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                id VARCHAR PRIMARY KEY,
                username VARCHAR UNIQUE NOT NULL,
                email VARCHAR UNIQUE,
                password_hash VARCHAR NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                revoked BOOLEAN NOT NULL DEFAULT FALSE
            );
            """)
    finally:
        connection.close()


@app.on_event("startup")
def startup() -> None:
    """Initialize local PoC database tables on backend startup."""
    init_auth_tables()


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=256)
    email: Optional[str] = Field(default=None, max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AuthUserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    is_active: bool
    email_verified: bool


class AuthResponse(BaseModel):
    status: str
    token: str
    user: AuthUserResponse


class LogoutResponse(BaseModel):
    status: str


# -----------------------------------------------------------------------------
# Auth helpers
# -----------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt. Bcrypt includes its own random salt."""
    password_bytes = password.encode("utf-8")
    password_hash = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
    return password_hash.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8"),
    )


def create_session(connection: duckdb.DuckDBPyConnection, user_id: str) -> str:
    token = secrets.token_urlsafe(48)
    created_at = now_utc()
    expires_at = created_at + timedelta(hours=SESSION_TTL_HOURS)

    connection.execute(
        """
        INSERT INTO auth_sessions (
            token,
            user_id,
            created_at,
            expires_at,
            revoked
        )
        VALUES (?, ?, ?, ?, FALSE)
        """,
        [token, user_id, created_at, expires_at],
    )

    return token


def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    prefix = "Bearer "

    if not authorization.startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )

    token = authorization[len(prefix) :].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    return token


def get_user_by_session_token(token: str) -> AuthUserResponse:
    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                u.id,
                u.username,
                u.email,
                u.is_active,
                u.email_verified,
                s.expires_at,
                s.revoked
            FROM auth_sessions s
            JOIN app_users u ON u.id = s.user_id
            WHERE s.token = ?
            LIMIT 1
            """,
            [token],
        ).fetchone()
    finally:
        connection.close()

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    user_id, username, email, is_active, email_verified, expires_at, revoked = result

    if revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session revoked",
        )

    expires_at_utc = expires_at

    if expires_at_utc.tzinfo is None:
        expires_at_utc = expires_at_utc.replace(tzinfo=timezone.utc)

    if expires_at_utc <= now_utc():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
        )

    if not is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User inactive",
        )

    return AuthUserResponse(
        id=user_id,
        username=username,
        email=email,
        is_active=is_active,
        email_verified=email_verified,
    )


# -----------------------------------------------------------------------------
# System endpoints
# -----------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return a minimal health response for smoke tests."""
    runtime_path = get_runtime_data_path()

    return {
        "status": "ok",
        "app_env": settings.app_env,
        "runtime_data_path": str(runtime_path),
        "duckdb_path": settings.duckdb_path,
    }


@app.get("/health/duckdb", tags=["System"])
def duckdb_health_check() -> dict[str, str]:
    """Verify that the configured DuckDB path can be opened."""
    duckdb_path = get_duckdb_path()
    connection = duckdb.connect(str(duckdb_path))

    try:
        connection.execute("select 1")
    finally:
        connection.close()

    return {
        "status": "ok",
        "duckdb_path": str(duckdb_path),
    }


@app.get("/config/runtime", tags=["System"])
def runtime_config() -> dict[str, str | bool]:
    """Return non-secret runtime configuration for PoC verification."""
    return {
        "app_env": settings.app_env,
        "log_level": settings.log_level,
        "runtime_data_path": settings.runtime_data_path,
        "duckdb_path": settings.duckdb_path,
        "guldnas_duckdb_path": settings.guldnas_duckdb_path,
        "enable_audit_log": settings.enable_audit_log,
        "enable_risk_output": settings.enable_risk_output,
    }


# -----------------------------------------------------------------------------
# Auth endpoints
# -----------------------------------------------------------------------------


@app.post("/auth/register", response_model=AuthResponse, tags=["Auth"])
def register_user(payload: RegisterRequest) -> AuthResponse:
    """Create a persistent local PoC user and return a session token."""
    username = payload.username.strip().lower()
    email = payload.email.strip().lower() if payload.email else None

    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is required",
        )

    user_id = str(uuid.uuid4())
    created_at = now_utc()
    password_hash = hash_password(payload.password)

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO app_users (
                id,
                username,
                email,
                password_hash,
                is_active,
                email_verified,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, TRUE, FALSE, ?, ?)
            """,
            [
                user_id,
                username,
                email,
                password_hash,
                created_at,
                created_at,
            ],
        )

        token = create_session(connection, user_id)

    except Exception as exc:
        message = str(exc).lower()

        if "unique" in message or "duplicate" in message or "constraint" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username or email already exists",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not register user: {exc}",
        ) from exc

    finally:
        connection.close()

    return AuthResponse(
        status="ok",
        token=token,
        user=AuthUserResponse(
            id=user_id,
            username=username,
            email=email,
            is_active=True,
            email_verified=False,
        ),
    )


@app.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
def login_user(payload: LoginRequest) -> AuthResponse:
    """Login a persistent PoC user and return a session token."""
    username = payload.username.strip().lower()

    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                id,
                username,
                email,
                password_hash,
                is_active,
                email_verified
            FROM app_users
            WHERE username = ?
            LIMIT 1
            """,
            [username],
        ).fetchone()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        user_id, stored_username, email, password_hash, is_active, email_verified = (
            result
        )

        if not is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User inactive",
            )

        if not verify_password(payload.password, password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        token = create_session(connection, user_id)

    finally:
        connection.close()

    return AuthResponse(
        status="ok",
        token=token,
        user=AuthUserResponse(
            id=user_id,
            username=stored_username,
            email=email,
            is_active=is_active,
            email_verified=email_verified,
        ),
    )


@app.get("/auth/me", response_model=AuthUserResponse, tags=["Auth"])
def auth_me(authorization: Optional[str] = Header(default=None)) -> AuthUserResponse:
    """Return the current authenticated PoC user."""
    token = extract_bearer_token(authorization)
    return get_user_by_session_token(token)


@app.post("/auth/logout", response_model=LogoutResponse, tags=["Auth"])
def logout_user(authorization: Optional[str] = Header(default=None)) -> LogoutResponse:
    """Revoke the current PoC session token."""
    token = extract_bearer_token(authorization)

    connection = get_connection()

    try:
        connection.execute(
            """
            UPDATE auth_sessions
            SET revoked = TRUE
            WHERE token = ?
            """,
            [token],
        )
    finally:
        connection.close()

    return LogoutResponse(status="ok")
