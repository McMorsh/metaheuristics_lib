"""Тесты демонстрационных скриптов."""

import csv
import importlib
import json
import logging
from dataclasses import fields
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


def test_benchmark_propagates_target() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=5,
        agents=5,
        runs=1,
        seed=0,
        target=100.0,
    )

    assert all(row.stop_reason == "target 1/1" for row in rows)
    assert all(row.nfev == 5 for row in rows)


def test_benchmark_propagates_max_fev() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=5,
        agents=5,
        runs=1,
        seed=0,
        max_fev=5,
    )

    assert all(row.stop_reason == "max_fev 1/1" for row in rows)
    assert all(row.nfev == 5 for row in rows)


def test_benchmark_computes_best_at_budget() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=10,
        agents=5,
        runs=1,
        seed=0,
        budget=10,
    )

    assert all(row.budget_reached == "1/1" for row in rows)
    for row in rows:
        assert row.best_at_budget is not None
        assert row.best_at_budget >= row.best

    text = benchmark_example.format_rows(rows)
    assert "f(x)@budget" in text
    assert "budget" in text


def test_benchmark_budget_above_total_returns_none() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=2,
        agents=5,
        runs=1,
        seed=0,
        budget=10000,
    )

    assert all(row.best_at_budget is None for row in rows)
    assert all(row.budget_reached == "0/1" for row in rows)


def test_benchmark_budget_below_population_returns_none() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.0, 5.0),
        dim=2,
        iterations=2,
        agents=5,
        runs=1,
        seed=0,
        budget=3,
    )

    assert all(row.best_at_budget is None for row in rows)
    assert all(row.budget_reached == "0/1" for row in rows)
    assert "0/1" in benchmark_example.format_rows(rows)


def test_benchmark_rejects_zero_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        benchmark_example.run_benchmark(
            sphere,
            (-5.0, 5.0),
            dim=2,
            iterations=1,
            agents=5,
            runs=1,
            seed=0,
            budget=0,
        )


def test_benchmark_logs_iterations_when_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="metaheuristics_lib.core.runner",
    ):
        rows = benchmark_example.run_benchmark(
            sphere,
            (-5.0, 5.0),
            dim=2,
            iterations=1,
            agents=5,
            runs=1,
            seed=0,
            iteration_info=True,
        )

    assert rows
    assert "Iteration 1 end" in caplog.text


def test_benchmark_aggregates_mixed_stop_reasons() -> None:
    rows = benchmark_example.run_benchmark(
        sphere,
        (-5.12, 5.12),
        dim=2,
        iterations=2,
        agents=5,
        runs=2,
        seed=0,
        max_fev=5,
        target=5.0,
    )

    assert all(row.stop_reason == "max_fev 1/2, target 1/2" for row in rows)


def test_format_rows_prints_stop_reason() -> None:
    rows = [
        benchmark_example.BenchmarkRow(
            name="Stopped",
            best=1.0,
            nfev=10,
            elapsed=0.5,
            stop_reason="target 2/3, max_fev 1/3",
        ),
        benchmark_example.BenchmarkRow(
            name="Finished",
            best=2.0,
            nfev=20,
            elapsed=1.0,
            stop_reason=None,
        ),
    ]

    lines = benchmark_example.format_rows(rows).splitlines()

    assert "остановка" in lines[0]
    assert "target 2/3, max_fev 1/3" in lines[2]
    assert lines[3].endswith(" " + "-")
    assert len({len(line) for line in lines}) == 1


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


def test_run_experiments_supports_multiple_problems_and_dims() -> None:
    records = benchmark_example.run_experiments(
        ["sphere", "ackley"],
        [2, 3],
        iterations=1,
        agents=5,
        runs=1,
        seed=0,
    )

    assert len(records) == (2 * 2 * len(benchmark_example.ALGORITHMS))
    assert {(record.problem, record.dim) for record in records} == {
        ("sphere", 2),
        ("sphere", 3),
        ("ackley", 2),
        ("ackley", 3),
    }

    rows = benchmark_example.summarize_records(records)
    text = benchmark_example.format_rows(rows)

    assert len(rows) == len(records)
    assert "Задача" in text
    assert "dim" in text


def test_run_experiments_rejects_fixed_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dim=2"):
        benchmark_example.run_experiments(
            ["himmelblau"],
            [3],
            iterations=1,
            agents=5,
            runs=1,
            seed=0,
        )


def _make_record(
    best: float,
    *,
    seed: int = 0,
    nfev: int = 5,
    budget: int | None = None,
    best_at_budget: float | None = None,
) -> benchmark_example.RunRecord:
    return benchmark_example.RunRecord(
        algorithm="A",
        problem="sphere",
        dim=2,
        seed=seed,
        best=best,
        nfev=nfev,
        stop_reason=None,
        elapsed=0.1,
        budget=budget,
        best_at_budget=best_at_budget,
    )


def test_summarize_records_computes_statistics() -> None:
    records = [
        _make_record(1.0, seed=0),
        _make_record(2.0, seed=1),
        _make_record(3.0, seed=2),
    ]

    rows = benchmark_example.summarize_records(records, target=2.0)

    assert len(rows) == 1
    row = rows[0]
    assert row.best == pytest.approx(2.0)
    assert row.median == pytest.approx(2.0)
    assert row.std == pytest.approx(1.0)
    assert row.success_rate == pytest.approx(2 / 3)
    assert row.runs == 3


def test_summarize_records_aggregates_budget() -> None:
    records = [
        _make_record(
            0.5,
            seed=0,
            nfev=10,
            budget=10,
            best_at_budget=1.0,
        ),
        _make_record(
            2.0,
            seed=1,
            nfev=5,
            budget=10,
            best_at_budget=None,
        ),
    ]

    rows = benchmark_example.summarize_records(records)

    assert len(rows) == 1
    assert rows[0].best_at_budget == pytest.approx(1.0)
    assert rows[0].budget_reached == "1/2"


def test_format_rows_shows_problem_and_statistics() -> None:
    row = benchmark_example.BenchmarkRow(
        name="A",
        best=2.0,
        nfev=5,
        elapsed=0.1,
        problem="sphere",
        dim=2,
        median=2.0,
        std=1.0,
        success_rate=0.5,
        runs=3,
    )

    text = benchmark_example.format_rows([row])

    assert "Задача" in text
    assert "медиана" in text
    assert "std" in text
    assert "success" in text


def test_run_fields_match_record_fields() -> None:
    record_fields = {
        field.name for field in fields(benchmark_example.RunRecord)
    }

    assert set(benchmark_example.RUN_FIELDS) == record_fields


def test_save_records_json(tmp_path: Path) -> None:
    records = [_make_record(1.0)]
    path = tmp_path / "runs.json"

    benchmark_example.save_records(records, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["algorithm"] == "A"
    assert payload[0]["best"] == pytest.approx(1.0)


def test_save_records_csv(tmp_path: Path) -> None:
    records = [_make_record(1.0)]
    path = tmp_path / "runs.csv"

    benchmark_example.save_records(records, path)

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["algorithm"] == "A"
    assert float(rows[0]["best"]) == pytest.approx(1.0)


def test_save_records_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Неподдерживаемый"):
        benchmark_example.save_records([], tmp_path / "runs.txt")
