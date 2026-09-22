# ============================================================
# tests/test_rag.py —— RAG 单测（只测纯函数，不依赖 ChromaDB）
# ============================================================

from agent.rag import chunk_text, rewrite_query


class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []

    def test_short_paragraph(self):
        text = "这是一段短文本。"
        chunks = chunk_text(text, max_len=100)
        assert chunks == [text]

    def test_long_paragraph_split(self):
        # 无标点的长文本，会走按句子兜底逻辑
        text = "a" * 500
        chunks = chunk_text(text, max_len=100)
        assert len(chunks) >= 5
        assert all(len(c) <= 100 for c in chunks)

    def test_multi_paragraph(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunk_text(text, max_len=100)
        assert len(chunks) == 3


class TestRewriteQuery:
    def test_fallback_on_failure(self, monkeypatch):
        """LLM 调用失败时应回退原 query，不抛异常。"""
        def boom(*a, **kw):
            raise RuntimeError("mock failure")

        monkeypatch.setattr("agent.llm_client.get_client", boom)
        assert rewrite_query("北京天气") == "北京天气"
