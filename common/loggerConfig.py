from loguru import logger
from pathlib import Path
from datetime import datetime
import sys

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = logs_dir / f"run_{timestamp}.log"

# Clear default loguru handler
logger.remove()

# Add colorful console logger
logger.add(
    sys.stdout,
    level="TRACE",  # logs everything
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=True,
)

# Add log file output (timestamped)
logger.add(
    str(log_file),
    level="TRACE",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    ),
    backtrace=True,
    diagnose=True
)

# Expose logger to be imported
__all__ = ["logger"]
