# ============================================================
# logging_config.py —— 统一日志配置（保留在项目根目录）
# ============================================================

import logging
from logging.handlers import RotatingFileHandler

from agent.config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)


def setup_logging(console: bool = False, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    # 清空已有 handler，避免重复注册
    for h in root.handlers[:]:
        root.removeHandler(h)

    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = RotatingFileHandler(
        LOG_DIR / "agent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(ch)

    # 压制第三方库 INFO 日志
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)