"""Тесты публичного фасада библиотеки."""

import tomllib
from pathlib import Path

import pytest

import metaheuristics_lib


def test_facade_exports_declared_names() -> None:
    assert len(metaheuristics_lib.__all__) == len(
        set(metaheuristics_lib.__all__)
    )
    for name in metaheuristics_lib.__all__:
        assert hasattr(metaheuristics_lib, name), name


def test_version_matches_pyproject() -> None:
    if metaheuristics_lib.__version__ == "0.0.0":
        pytest.skip("пакет не установлен")

    project_root = Path(__file__).resolve().parents[1]
    pyproject = project_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metaheuristics_lib.__version__ == data["project"]["version"]


def test_facade_runs_algorithm() -> None:
    algorithm = metaheuristics_lib.WhaleOptimizationAlgorithm(
        metaheuristics_lib.sphere,
        2,
        (-1.0, 1.0),
        agents=4,
        max_iter=1,
        seed=42,
    )
    with metaheuristics_lib.Runner(
        algorithm,
        iteration_info=False,
    ) as runner:
        result, history, _, _ = runner.run()

    assert result["minimum_value"] == metaheuristics_lib.sphere(
        result["minimum_x"]
    )
    assert len(history) == 2
