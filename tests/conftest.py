# ============================================================
# tests/conftest.py —— pytest 共享 fixture
# ============================================================

import os
import sys
from pathlib import Path

import pytest

# 保证测试能 import 项目根目录的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """把 DB_PATH 指到临时目录，避免污染真实 cities.db。"""
    from agent import config
    test_db = tmp_path / "test_cities.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)

    # 因为 tools.py 是 `from agent.config import DB_PATH`，
    # 需要在 tools 模块里也 patch 一次
    from agent import tools
    monkeypatch.setattr(tools, "DB_PATH", test_db)

    tools.init_db()
    return test_db


@pytest.fixture
def tmp_docs(tmp_path, monkeypatch):
    """把 DOCS_DIR 指到临时目录，避免读真实 docs。"""
    from agent import config
    from agent import rag
    test_docs = tmp_path / "docs"
    test_docs.mkdir()
    monkeypatch.setattr(config, "DOCS_DIR", test_docs)
    monkeypatch.setattr(rag, "DOCS_DIR", test_docs)
    return test_docs