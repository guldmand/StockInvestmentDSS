"""Configuration for the StockInvestmentDSS PoC backend."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_env: str = "local"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    runtime_data_path: str = "./runtime-data"
    duckdb_path: str = "./runtime-data/market_research.duckdb"
    guldnas_duckdb_path: str = (
        "/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb"
    )
    enable_audit_log: bool = True
    enable_risk_output: bool = True
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
