"""Тесты настройки логгеров."""

import logging
from pathlib import Path

import pytest

from utils.logger import get_logger


def _clear_handlers(name: str) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()


def test_get_logger_is_idempotent() -> None:
    name = "test_idempotent"
    try:
        logger = get_logger(name)
        handler_count = len(logger.handlers)
        second = get_logger(name)
        assert second is logger
        assert len(logger.handlers) == handler_count
    finally:
        _clear_handlers(name)


def test_file_handler_writes_messages(tmp_path: Path) -> None:
    name = "test_file_handler"
    path = tmp_path / "app.log"
    try:
        logger = get_logger(
            name,
            log_to_file=True,
            filename=path,
        )
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        assert path.exists()
        assert "hello" in path.read_text(encoding="utf-8")
    finally:
        _clear_handlers(name)


def test_invalid_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="уровень"):
        get_logger("test_invalid_level", level="UNKNOWN")


def test_invalid_max_bytes_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        get_logger("test_invalid_bytes", max_bytes=0)


def test_invalid_backup_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="backup_count"):
        get_logger("test_invalid_backup", backup_count=-1)


def test_integer_level_is_accepted() -> None:
    name = "test_integer_level"
    try:
        logger = get_logger(name, level=logging.DEBUG)
        assert logger.level == logging.DEBUG
    finally:
        _clear_handlers(name)
