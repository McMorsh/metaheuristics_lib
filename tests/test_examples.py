"""Тесты демонстрационных скриптов."""

import importlib
from pathlib import Path

from examples import test as serial_example
from examples import test_mp as parallel_example


def test_serial_example_writes_plots(tmp_path: Path) -> None:
    serial_example.test_alg(
        iterations=2,
        agents=5,
        output_dir=tmp_path,
    )
    serial_example.multi_run_test(
        iterations=2,
        agents=5,
        runs=2,
        output_dir=tmp_path,
    )

    assert (tmp_path / "csa.png").exists()
    assert (tmp_path / "woa_runs.png").exists()


def test_parallel_example_writes_plot(tmp_path: Path) -> None:
    parallel_example.test_alg(
        iterations=2,
        agents=5,
        process_counts=[1],
        output_dir=tmp_path,
    )

    assert (tmp_path / "woa_speedup.png").exists()


def test_mpi_example_import_is_lazy() -> None:
    module = importlib.import_module("examples.run_mpi")
    assert module is not None
