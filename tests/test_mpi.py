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
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")


def _terminate(process: subprocess.Popen[str]) -> None:
    """Завершает процесс и его группу, если это возможно."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


@pytest.mark.integration
def test_mpi_example_runs_with_two_ranks() -> None:
    if MPIEXEC is None:
        pytest.skip("mpiexec не найден")
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("mpi4py не установлен")

    process = subprocess.Popen(
        [
            MPIEXEC,
            "-n",
            "2",
            sys.executable,
            "-m",
            "examples.run_mpi",
            "--iterations",
            "2",
            "--agents",
            "6",
        ],
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
        pytest.fail(f"MPI-пример не завершился за 120 секунд:\n{stderr}")

    assert process.returncode == 0, stderr
    assert "Лучшее значение функции" in stdout
