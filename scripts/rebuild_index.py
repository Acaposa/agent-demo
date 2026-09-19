# ============================================================
# scripts/rebuild_index.py —— 手动全量重建 RAG 索引
# ------------------------------------------------------------
# 用法（项目根目录执行）：
#   python -m scripts.rebuild_index
# ============================================================

import logging
import sys
from pathlib import Path

# 允许从项目根目录直接 python scripts/rebuild_index.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.rag import rebuild_all, sync_index
from logging_config import setup_logging


def main():
    setup_logging(console=True)
    logger = logging.getLogger("rebuild")

    logger.info("开始增量同步...")
    collection = sync_index()
    logger.info(f"当前集合条目数: {collection.count()}")

    logger.info("开始全量重建...")
    rebuild_all()
    logger.info("完成。")


if __name__ == "__main__":
    main()