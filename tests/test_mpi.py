"""Интеграционные тесты MPI."""

import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "mpi_worker.py"
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")

SCENARIOS = [
    "abc",
    "csa",
    "epo",
    "gwo",
    "woa",
    "error",
    "empty",
]


def _require_mpi() -> None:
    """Пропускает тест, если MPI недоступен."""
    if MPIEXEC is None:
        pytest.skip("mpiexec не найден")
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("mpi4py не установлен")


def _terminate(process: subprocess.Popen[str]) -> None:
    """Завершает процесс и его группу, если это возможно."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


def _run_mpi(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Запускает команду через mpiexec с двумя рангами."""
    assert MPIEXEC is not None
    command = [MPIEXEC, "-n", "2", sys.executable, *arguments]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        _terminate(process)
        stdout, stderr = process.communicate()
        pytest.fail(f"MPI-команда не завершилась за 120 секунд:\n{stderr}")

    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


@pytest.mark.integration
def test_mpi_example_runs_with_two_ranks() -> None:
    _require_mpi()

    result = _run_mpi(
        [
            "-m",
            "examples.run_mpi",
            "--iterations",
            "2",
            "--agents",
            "6",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "Лучшее значение функции" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_mpi_scenario_passes(scenario: str) -> None:
    _require_mpi()

    result = _run_mpi([str(WORKER), scenario])

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
