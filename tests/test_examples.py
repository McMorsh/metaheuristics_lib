"""Тесты демонстрационных скриптов."""

import importlib
from pathlib import Path

import pytest

from examples import benchmark as benchmark_example
from examples import run_mp as parallel_example
from examples import run_serial as serial_example
from metaheuristics_lib.problems.continuous import sphere


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


def test_benchmark_returns_all_algorithms() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=2,
        agents=5,
        runs=1,
        seed=0,
    )

    assert len(rows) == len(benchmark_example.ALGORITHMS)
    assert all(row.nfev > 0 for row in rows)
    assert all(row.elapsed >= 0.0 for row in rows)

    text = benchmark_example.format_rows(rows)
    assert "WhaleOptimizationAlgorithm" in text
    assert "nfev" in text


def test_benchmark_rejects_zero_runs() -> None:
    with pytest.raises(ValueError, match="runs"):
        benchmark_example.run_benchmark(
            sphere,
            (-5.0, 5.0),
            dim=2,
            iterations=1,
            agents=5,
            runs=0,
            seed=0,
        )
