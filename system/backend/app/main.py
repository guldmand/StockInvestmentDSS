"""FastAPI entrypoint for the StockInvestmentDSS PoC backend."""

from pathlib import Path
import duckdb
from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title="StockInvestmentDSS API",
    description="PoC API for data-driven stock investment decision support.",
    version="0.1.0",
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return a minimal health response for smoke tests."""
    runtime_path = Path(settings.runtime_data_path)
    runtime_path.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "runtime_data_path": settings.runtime_data_path,
        "duckdb_path": settings.duckdb_path,
    }


@app.get("/health/duckdb", tags=["System"])
def duckdb_health_check() -> dict[str, str]:
    """Verify that the configured DuckDB path can be opened."""
    duckdb_path = Path(settings.duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(duckdb_path))
    connection.execute("select 1")
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
