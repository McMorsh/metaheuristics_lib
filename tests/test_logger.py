"""Тесты настройки логгеров."""

import logging
from pathlib import Path

import pytest

from utils.logger import (
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
    get_logger,
)


def _clear_handlers(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.close()
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


def test_repeated_get_logger_updates_handler_level() -> None:
    name = "test_update_handler_level"
    try:
        logger = get_logger(name, level=logging.INFO)
        console = next(
            handler
            for handler in logger.handlers
            if handler.name == CONSOLE_HANDLER_NAME
        )
        assert console.level == logging.INFO

        get_logger(name, level=logging.DEBUG)

        assert logger.level == logging.DEBUG
        assert console.level == logging.DEBUG
    finally:
        _clear_handlers(name)


def test_repeated_get_logger_updates_file_handler_level(
    tmp_path: Path,
) -> None:
    name = "test_update_file_handler_level"
    path = tmp_path / "app.log"
    try:
        logger = get_logger(
            name,
            level=logging.INFO,
            log_to_file=True,
            filename=path,
        )
        file_handler = next(
            handler
            for handler in logger.handlers
            if handler.name == FILE_HANDLER_NAME
        )
        assert file_handler.level == logging.INFO

        get_logger(
            name,
            level=logging.DEBUG,
            log_to_file=True,
            filename=path,
        )

        assert file_handler.level == logging.DEBUG
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


def test_string_level_is_accepted() -> None:
    name = "test_string_level"
    try:
        logger = get_logger(name, level="debug")
        assert logger.level == logging.DEBUG
        console = next(
            handler
            for handler in logger.handlers
            if handler.name == CONSOLE_HANDLER_NAME
        )
        assert console.level == logging.DEBUG
    finally:
        _clear_handlers(name)


def test_get_logger_disables_propagation() -> None:
    name = "test_propagation"
    try:
        logger = get_logger(name)
        assert logger.propagate is False
    finally:
        _clear_handlers(name)
