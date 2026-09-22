# ---------- 记忆压缩 ----------

def test_compress_triggers_when_exceeding_threshold(monkeypatch):
    """消息数超阈值时触发压缩，调用 _llm_summarize。"""
    from agent import memory

    calls = []
    monkeypatch.setattr(
        memory, "_llm_summarize",
        lambda prev, msgs: (calls.append((prev, len(msgs))), "模拟摘要")[1],
    )

    s = memory.AgentSession()
    # 每轮 2 条，加 10 轮 = 20 条非 system，超过 COMPRESS_THRESHOLD(16)
    for i in range(10):
        s.add_user(f"问题{i}")
        s.add_assistant(f"回答{i}")

    assert len(calls) >= 1, "应至少触发一次压缩"
    assert s.summary == "模拟摘要"
    # 压缩后消息应明显少于 21 条（1 system + 20 对话）
    assert len(s.messages) < 21


def test_compress_fail_keeps_messages(monkeypatch):
    """压缩失败（LLM 返回 None）时，messages 保持原状，不抛异常。"""
    from agent import memory

    monkeypatch.setattr(memory, "_llm_summarize", lambda prev, msgs: None)

    s = memory.AgentSession()
    for i in range(10):
        s.add_user(f"问题{i}")
        s.add_assistant(f"回答{i}")

    assert s.summary is None
    # 因为压缩失败不裁切，messages 保留全部
    assert len(s.messages) == 1 + 20


def test_cut_index_lands_on_user_message(monkeypatch):
    """切点必须落在 user 消息，保证 tool_calls 链完整。"""
    from agent import memory

    captured = {}

    def fake_summarize(prev, msgs):
        captured["first_role"] = msgs[0].get("role")
        return "摘要"

    monkeypatch.setattr(memory, "_llm_summarize", fake_summarize)

    s = memory.AgentSession()
    for i in range(10):
        s.add_user(f"问题{i}")
        s.add_assistant(f"回答{i}")

    assert captured.get("first_role") == "user", "压缩起点必须是 user"


def test_session_persistence(tmp_path, monkeypatch):
    """会话保存后能从 SQLite 恢复。"""
    from agent import memory, session_store

    monkeypatch.setattr(session_store, "SESSIONS_DB", tmp_path / "sessions.db")

    s = memory.AgentSession(session_id="test-1")
    s.add_user("北京天气怎么样")
    s.add_assistant("北京今天晴，28°C")
    s.save()

    s2 = memory.AgentSession.load_or_create("test-1")
    assert s2.session_id == "test-1"
    assert len(s2.messages) >= 3
    assert any("北京天气" in (m.get("content") or "") for m in s2.messages)


def test_session_load_nonexistent_creates_new(tmp_path, monkeypatch):
    """加载不存在的 session_id 应返回空会话。"""
    from agent import memory, session_store

    monkeypatch.setattr(session_store, "SESSIONS_DB", tmp_path / "sessions.db")

    s = memory.AgentSession.load_or_create("does-not-exist")
    assert s.session_id == "does-not-exist"
    assert len(s.messages) == 1   # 只有 system


def test_session_without_id_skips_save(tmp_path, monkeypatch):
    """无 session_id 时 save() 静默跳过，不创建 DB 文件。"""
    from agent import memory, session_store

    db_path = tmp_path / "should_not_exist.db"
    monkeypatch.setattr(session_store, "SESSIONS_DB", db_path)

    s = memory.AgentSession()   # 不传 session_id
    s.add_user("test")
    s.save()

    assert not db_path.exists()