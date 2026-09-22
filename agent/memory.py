# ============================================================
# agent/memory.py —— 会话记忆（滑动窗口 + LLM 摘要压缩 + 持久化）
# ============================================================

import logging

from agent.config import (
    SYSTEM_PROMPT,
    COMPRESS_THRESHOLD,
    COMPRESS_KEEP_RECENT,
    MAX_SUMMARY_CHARS,
)

logger = logging.getLogger("agent")


class AgentSession:
    """一次会话的消息容器。多轮对话时复用同一个实例。"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, session_id: str | None = None):
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.summary: str | None = None
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._maybe_compress()

    def add_assistant(self, content: str | None, tool_calls: list | None = None) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def reset(self) -> None:
        """清空对话和摘要，只留 system。"""
        self.summary = None
        self.messages = [{"role": "system", "content": self.system_prompt}]

    # ---------- 持久化 ----------

    def save(self) -> None:
        """持久化到 SQLite。无 session_id 时静默跳过。"""
        if not self.session_id:
            return
        from agent.session_store import save_session
        save_session(self.session_id, self.summary, self.messages)

    @classmethod
    def load_or_create(
        cls,
        session_id: str,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> "AgentSession":
        """按 session_id 恢复会话；不存在则新建。"""
        session = cls(system_prompt, session_id=session_id)
        from agent.session_store import load_session
        data = load_session(session_id)
        if data:
            summary, messages = data
            session.summary = summary
            session.messages = messages
            logger.info(f"已恢复会话 {session_id}（{len(messages)} 条消息）")
        return session

    # ---------- 摘要压缩 ----------

    def _maybe_compress(self) -> None:
        non_system = len(self.messages) - 1
        if non_system < COMPRESS_THRESHOLD:
            return

        cut = self._find_cut_index()
        if cut is None:
            return

        to_summarize = self.messages[1:cut]
        if not to_summarize:
            return

        logger.info(f"触发记忆压缩：压缩 {len(to_summarize)} 条旧消息")
        new_summary = _llm_summarize(self.summary, to_summarize)
        if new_summary is None:
            logger.warning("记忆压缩失败，messages 保持原状")
            return

        self.summary = new_summary
        self._rebuild(cut)
        logger.info(
            f"压缩完成：messages 从 {non_system + 1} 条缩减为 {len(self.messages)} 条"
        )

    def _find_cut_index(self) -> int | None:
        """找压缩切点。切点必须是 user 消息，保证工具调用链不被切断。"""
        for idx in range(len(self.messages) - 1, 0, -1):
            if self.messages[idx].get("role") != "user":
                continue
            tail_count = len(self.messages) - idx
            if tail_count >= COMPRESS_KEEP_RECENT:
                return idx
        return None

    def _rebuild(self, cut_index: int) -> None:
        kept = self.messages[cut_index:]
        new_messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if self.summary:
            new_messages.append({
                "role": "system",
                "content": (
                    "【早前对话摘要】以下是本次会话之前的内容摘要，"
                    "供你理解上下文，不要把它当成用户的新问题：\n"
                    f"{self.summary}"
                ),
            })
        new_messages.extend(kept)
        self.messages = new_messages


def _llm_summarize(prev_summary: str | None, messages: list[dict]) -> str | None:
    """把 messages 压缩成摘要；prev_summary 非空时与其合并（滚动摘要）。"""
    from agent.llm_client import get_client

    lines: list[str] = []
    if prev_summary:
        lines.append(f"【已有摘要】\n{prev_summary}\n")
    lines.append("【新增对话】")
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            if content:
                lines.append(f"助手：{content}")
            tcs = m.get("tool_calls")
            if tcs:
                names = [tc.get("function", {}).get("name", "?") for tc in tcs]
                lines.append(f"（助手调用了工具：{', '.join(names)}）")
        elif role == "tool":
            lines.append(f"（工具结果：{content[:200]}）")

    text = "\n".join(lines)
    prompt = (
        "请把下面这段多轮对话压缩成简洁的中文摘要。要求：\n"
        "1) 保留用户问过什么、助手用了哪些工具、得出了什么结论；\n"
        "2) 不要臆测未发生的内容，不要罗列无关细节；\n"
        f"3) 摘要控制在 {MAX_SUMMARY_CHARS} 字以内。\n\n"
        f"{text}"
    )

    try:
        resp = get_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return summary or None
    except Exception as e:
        logger.warning(f"摘要 LLM 调用失败：{e}")
        return None