from __future__ import annotations

import logging
from pathlib import Path

from stock_investment_dss.utilities.paths import LOGS_DIRECTORY, RunPaths

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_system_logger(log_level: str = "INFO") -> logging.Logger:
    """
    System/debug logger.

    Used for:
    - startup errors
    - data pipeline failures
    - Python exceptions before a run exists
    """
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stock_investment_dss.system")
    logger.setLevel(log_level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    system_file_handler = logging.FileHandler(
        LOGS_DIRECTORY / "system.log",
        encoding="utf-8",
    )
    system_file_handler.setLevel(logging.INFO)
    system_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(
        LOGS_DIRECTORY / "errors.log",
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())
    console_handler.setFormatter(formatter)

    logger.addHandler(system_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_run_logger(run_paths: RunPaths, log_level: str = "INFO") -> logging.Logger:
    """
    Run-specific logger.

    Used for:
    - one concrete training/simulation run
    - run.log
    - run-level errors.log
    """
    run_paths.logs_directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"stock_investment_dss.run.{run_paths.run_id}")
    logger.setLevel(log_level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    run_file_handler = logging.FileHandler(
        run_paths.logs_directory / "run.log",
        encoding="utf-8",
    )
    run_file_handler.setLevel(logging.INFO)
    run_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(
        run_paths.logs_directory / "errors.log",
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())
    console_handler.setFormatter(formatter)

    logger.addHandler(run_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_named_file_logger(
    logger_name: str,
    log_file: Path,
    log_level: str = "INFO",
) -> logging.Logger:
    """
    Optional logger for special cases, e.g. data_pipeline.log.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level.upper())
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
