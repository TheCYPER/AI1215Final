"""Logging setup for the pipeline."""

import logging
from pathlib import Path


def setup_logger(
    name: str = "main",
    log_file: str = "outputs/logs/main.log",
    level: str = "INFO",
) -> logging.Logger:
    """Configure the root logger so all modules share the same handlers."""
    root = logging.getLogger()

    if root.handlers:
        return logging.getLogger(name)

    root.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return logging.getLogger(name)
