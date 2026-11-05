"""Общие помощники для параллельных вариантов алгоритмов."""

import warnings


def resolve_processes(
    processes: int | None,
    process: int | None,
) -> int | None:
    """Разрешает старый и новый параметры числа процессов.

    Args:
        processes: актуальное имя параметра.
        process: устаревший синоним.

    Returns:
        Число процессов или ``None``.

    Raises:
        ValueError: если оба параметра заданы и различаются.
    """
    if process is not None:
        warnings.warn(
            "Параметр process устарел; используйте processes",
            DeprecationWarning,
            stacklevel=3,
        )
        if processes is not None and processes != process:
            raise ValueError(
                "Параметры processes и process противоречат друг другу"
            )
        return process
    return processes
