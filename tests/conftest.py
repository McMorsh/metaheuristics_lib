"""Общие настройки тестов."""

import logging
import multiprocessing
import os
import sys
import tempfile
from multiprocessing.context import BaseContext

import pytest


@pytest.fixture
def process_context() -> BaseContext | None:
    """Возвращает контекст процессов, доступный в окружении."""
    if (
        sys.platform.startswith("linux")
        and "fork" in multiprocessing.get_all_start_methods()
    ):
        return multiprocessing.get_context("fork")
    return None


def pytest_configure() -> None:
    """Настраивает matplotlib и уровень логов до сбора тестов."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        tempfile.mkdtemp(prefix="metaheuristics-mpl-"),
    )
    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("core.runner").setLevel(logging.WARNING)
