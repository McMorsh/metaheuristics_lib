"""Общие настройки тестов."""

import logging
import os
import tempfile


def pytest_configure() -> None:
    """Настраивает matplotlib и уровень логов до сбора тестов."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        tempfile.mkdtemp(prefix="metaheuristics-mpl-"),
    )
    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("core.runner").setLevel(logging.WARNING)
