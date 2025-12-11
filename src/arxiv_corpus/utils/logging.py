"""Logging configuration and utilities."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from arxiv_corpus.config import LoggingConfig

# Global console for rich output
console = Console()

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Set up logging configuration.

    Args:
        config: Logging configuration. If None, uses defaults.
    """
    if config is None:
        config = LoggingConfig()

    # Get the root logger for our package
    root_logger = logging.getLogger("arxiv_corpus")
    root_logger.setLevel(getattr(logging, config.level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, config.level))
    root_logger.addHandler(console_handler)

    # File handler if configured
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, config.level))
        file_handler.setFormatter(logging.Formatter(config.format))
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    if name not in _loggers:
        # Ensure it's under our package namespace
        if not name.startswith("arxiv_corpus"):
            name = f"arxiv_corpus.{name}"

        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


class ProgressLogger:
    """Context manager for logging progress of long-running operations."""

    def __init__(self, description: str, total: int | None = None) -> None:
        """Initialize progress logger.

        Args:
            description: Description of the operation.
            total: Total number of items (for progress bar).
        """
        self.description = description
        self.total = total
        self.logger = get_logger("progress")
        self._progress = None

    def __enter__(self) -> "ProgressLogger":
        """Start progress tracking."""
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
        ]

        if self.total is not None:
            columns.extend(
                [
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                ]
            )

        columns.append(TimeElapsedColumn())

        self._progress = Progress(*columns, console=console)
        self._progress.start()
        self._task_id = self._progress.add_task(self.description, total=self.total)

        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Stop progress tracking."""
        if self._progress:
            self._progress.stop()

    def update(self, advance: int = 1, description: str | None = None) -> None:
        """Update progress.

        Args:
            advance: Number of items completed.
            description: Optional new description.
        """
        if self._progress:
            kwargs: dict[str, int | str] = {"advance": advance}
            if description:
                kwargs["description"] = description
            self._progress.update(self._task_id, **kwargs)

    def log(self, message: str, level: str = "info") -> None:
        """Log a message while progress is running.

        Args:
            message: Message to log.
            level: Log level (debug, info, warning, error).
        """
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(message)
