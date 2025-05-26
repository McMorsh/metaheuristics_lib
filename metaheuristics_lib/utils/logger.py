'''
utils/logger.py

Модуль для настройки и получения логгеров в библиотеке метаэвристик.
Обеспечивает единообразное форматирование сообщений, вывод в консоль и (опционально)
запись в файл с ротацией.
'''
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import Optional, Union


def get_logger(
        name: str = __name__,
        level: Union[int, str] = logging.INFO,
        log_to_file: bool = False,
        filename: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
) -> Logger:
    """
    Создать и настроить логгер с указанным именем и уровнем.

    :param name: имя логгера (обычно __name__ модуля).
    :param level: уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :param log_to_file: если True, добавляется файловый обработчик.
    :param filename: имя файла для логов. Если None и log_to_file=True, используется '<name>.log'.
    :param max_bytes: максимальный размер файла до ротации (только для файлового хэндлера).
    :param backup_count: число резервных файлов при ротации.

    :return настроенный экземпляр logging.Logger

    Пример использования:
        logger = get_logger(__name__, level=logging.DEBUG, log_to_file=True, filename='app.log')
        logger.info("Запуск алгоритма...")
    """
    # Преобразуем уровень в числовой, если передана строка
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    # Получаем (или создаем) логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Если обработчики уже добавлены, не дублируем
    if not logger.handlers:
        # Создаем форматтер для всех хэндлеров
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Консольный хэндлер
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Файловый хэндлер с ротацией
        if log_to_file:
            if filename is None:
                filename = f"{name}.log"
            fh = RotatingFileHandler(
                filename, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
