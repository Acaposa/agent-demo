# ============================================================
# agent/rag.py —— RAG 检索模块（增量更新版）
# ------------------------------------------------------------
# 原 rag_store.py 迁移而来。路径统一从 agent.config 取。
# ============================================================

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from agent.config import CHROMA_DIR, DOCS_DIR, EMBEDDING_MODEL_PATH


logger = logging.getLogger("rag")

# 模块级缓存
_client = None
_collection = None
_file_hashes: dict[str, str] = {}
# 中文嵌入模型（惰性加载，首次调用时下载/加载）
_embedding_fn = None

# 文件哈希持久化路径（放在 chroma 目录下，便于统一清理）
HASH_FILE = CHROMA_DIR / "file_hashes.json"


def _load_hashes() -> None:
    """从磁盘加载文件哈希映射，使增量索引跨进程生效。"""
    global _file_hashes
    if HASH_FILE.exists():
        try:
            _file_hashes = json.loads(HASH_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[RAG] 读取哈希文件失败：{e}，回退为空映射")
            _file_hashes = {}
    else:
        _file_hashes = {}


def _save_hashes() -> None:
    """把当前文件哈希映射写回磁盘。"""
    try:
        HASH_FILE.parent.mkdir(exist_ok=True)
        HASH_FILE.write_text(
            json.dumps(_file_hashes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[RAG] 写入哈希文件失败：{e}")


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_PATH
        )
        logger.info(f"[RAG] 嵌入模型已加载: {EMBEDDING_MODEL_PATH}")
    return _embedding_fn


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """按段落切分长文本，段落超长则按句子切分；单句超长则硬切。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_len:
            chunks.append(p)
            continue

        # 按标点插入换行，再按行拼接
        for sep in ["。", "；", "，", "\n"]:
            p = p.replace(sep, sep + "\n")
        sentences = p.split("\n")

        buf = ""
        for s in sentences:
            if len(buf) + len(s) <= max_len:
                buf += s
                continue

            if buf:
                chunks.append(buf)
                buf = ""

            # 单句本身就超长：硬切
            while len(s) > max_len:
                chunks.append(s[:max_len])
                s = s[max_len:]

            buf = s

        if buf:
            chunks.append(buf)
    return chunks


def file_hash(fp: Path) -> str:
    return hashlib.md5(fp.read_bytes()).hexdigest()


def get_collection():
    """获取或创建 ChromaDB 文档集合（带缓存）。"""
    global _client, _collection

    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        _collection = _client.get_collection(
            "docs",
            embedding_function=get_embedding_fn()
        )
        logger.info("[RAG] 使用已有集合")
    except Exception:
        _collection = _client.create_collection(
            "docs",
            embedding_function=get_embedding_fn(),
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[RAG] 创建新集合（cosine 空间）")

    return _collection


def sync_index(force_rebuild: bool = False):
    """增量同步文档索引到 ChromaDB。"""
    global _collection, _file_hashes

    collection = get_collection()

    if force_rebuild:
        try:
            _client.delete_collection("docs")
        except Exception:
            pass
        _collection = _client.create_collection(
            "docs",
            embedding_function=get_embedding_fn(),
            metadata={"hnsw:space": "cosine"}
        )
        collection = _collection
        _file_hashes = {}
        _save_hashes()
        logger.info("[RAG] 强制全量重建")
    else:
        # 从磁盘恢复上次的哈希映射，避免每次新进程都全量重嵌入
        _load_hashes()

    if not DOCS_DIR.exists():
        logger.warning(f"[RAG] 文档目录不存在: {DOCS_DIR}")
        return collection

    current_files = {fp.name: fp for fp in DOCS_DIR.glob("*.md")}
    current_hashes = {name: file_hash(fp) for name, fp in current_files.items()}

    # 删除已移除的文件
    for old_name in list(_file_hashes.keys()):
        if old_name not in current_files:
            collection.delete(where={"source": old_name})
            logger.info(f"[RAG] 删除已移除文档: {old_name}")
            del _file_hashes[old_name]

    # 处理新增或修改的文件
    for name, fp in current_files.items():
        new_hash = current_hashes[name]
        if _file_hashes.get(name) == new_hash:
            continue

        collection.delete(where={"source": name})

        text = fp.read_text(encoding="utf-8")
        docs, ids, metadatas = [], [], []
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{fp.stem}_{i}")
            metadatas.append({"source": name, "chunk": i})

        if docs:
            collection.add(documents=docs, ids=ids, metadatas=metadatas)
            logger.info(f"[RAG] 更新文档: {name}，{len(docs)} 个片段")

        _file_hashes[name] = new_hash

    _save_hashes()
    return collection


def rewrite_query(query: str) -> str:
    """用 LLM 把用户问题改写成更适合向量检索的形式。
    失败或未启用时回退原 query（保证不阻塞主流程）。
    """
    from agent.config import ENABLE_QUERY_REWRITE
    if not ENABLE_QUERY_REWRITE:
        return query

    from agent.llm_client import get_client
    prompt = (
        "把下面的用户问题改写成一句简洁的中文检索查询。"
        "要求：保留关键实体和意图，去掉口语化词（如'帮我'、'请问'、'你知道'）。"
        "只输出改写后的查询本身，不要解释、不要引号。\n\n"
        f"原问题：{query}"
    )
    try:
        resp = get_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        if rewritten and rewritten != query:
            logger.info(f"[QueryRewrite] '{query}' → '{rewritten}'")
        return rewritten or query
    except Exception as e:
        logger.warning(f"[QueryRewrite] 失败：{e}，回退原 query")
        return query


def search_docs(query: str, top_k: int = 3, max_distance: float = 0.5) -> str:
    """
    检索本地知识库，返回相关文档片段。

    参数：
        query: 用户查询文本
        top_k: 返回的最相关片段数量
        max_distance: 最大距离阈值，超过则过滤
    """
    query = rewrite_query(query)
    collection = sync_index()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    logger.info(f"[检索] query='{query}' 距离: {[round(d, 2) for d in dists]}")

    parts = []
    for d, m, dist in zip(docs, metas, dists):
        if dist > max_distance:
            continue
        parts.append(f"[来源: {m['source']} 第{m['chunk']}段 | 距离:{dist:.2f}]\n{d}")

    if not parts:
        return "没有找到足够相关的文档内容"
    return "\n\n---\n\n".join(parts)


def rebuild_all():
    """强制全量重建文档索引。"""
    sync_index(force_rebuild=True)
    logger.info("[RAG] 全量重建完成")