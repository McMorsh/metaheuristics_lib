"""Настройка логгеров библиотеки метаэвристик."""

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_HANDLER_NAME = "metaheuristics-console"
FILE_HANDLER_NAME = "metaheuristics-file"


def _resolve_level(level: int | str) -> int:
    """Преобразует уровень логирования в число."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"Неизвестный уровень логирования: {level!r}")
    return resolved


def _update_handler_levels(logger: Logger, level: int) -> None:
    """Обновляет уровень обработчиков, созданных этой библиотекой."""
    managed_names = {CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME}
    for handler in logger.handlers:
        if handler.name in managed_names:
            handler.setLevel(level)


def get_logger(
    name: str = __name__,
    level: int | str = logging.INFO,
    log_to_file: bool = False,
    filename: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Logger:
    """Возвращает настроенный логгер.

    Повторный вызов с тем же именем не добавляет обработчики повторно,
    но обновляет их уровень.

    Args:
        name: имя логгера, обычно ``__name__`` модуля.
        level: числовой уровень или его имя.
        log_to_file: добавлять ли файловый обработчик.
        filename: путь к файлу логов. По умолчанию ``<name>.log``.
        max_bytes: размер файла до ротации.
        backup_count: число резервных файлов.

    Returns:
        Настроенный экземпляр ``logging.Logger``.

    Raises:
        ValueError: если уровень, размер файла или число резервных
            файлов некорректны.
    """
    if max_bytes < 1:
        raise ValueError(
            "Параметр max_bytes должен быть положительным, "
            f"получено {max_bytes}"
        )
    if backup_count < 0:
        raise ValueError(
            "Параметр backup_count не может быть отрицательным, "
            f"получено {backup_count}"
        )

    resolved_level = _resolve_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False
    _update_handler_levels(logger, resolved_level)

    formatter = logging.Formatter(
        fmt=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    if not any(
        handler.name == CONSOLE_HANDLER_NAME for handler in logger.handlers
    ):
        console = logging.StreamHandler()
        console.name = CONSOLE_HANDLER_NAME
        console.setLevel(resolved_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if not log_to_file:
        return logger

    if not any(
        handler.name == FILE_HANDLER_NAME for handler in logger.handlers
    ):
        path = Path(filename) if filename is not None else Path(f"{name}.log")
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.name = FILE_HANDLER_NAME
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
