# ============================================================
# agent/memory.py —— 会话记忆（最小实现）
# ------------------------------------------------------------
# 负责维护单次会话的 messages 列表。
# 现在只做滑动窗口截断，P2 会加摘要和跨会话持久化。
# ============================================================

from agent.config import SYSTEM_PROMPT


# 滑动窗口：保留最近 N 轮 user/assistant 消息（不含 system）
MAX_HISTORY_TURNS = 10


class AgentSession:
    """一次会话的消息容器。多轮对话时复用同一个实例。"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str | None, tool_calls: list | None = None) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def _trim(self) -> None:
        """保留 system + 最近 MAX_HISTORY_TURNS 轮（每轮约 2 条）。"""
        system = self.messages[0]
        rest = self.messages[1:]
        # 粗略保留：不超过 2 * MAX_HISTORY_TURNS 条
        limit = 2 * MAX_HISTORY_TURNS
        if len(rest) > limit:
            self.messages = [system] + rest[-limit:]

    def reset(self) -> None:
        """清空对话，只留 system。"""
        self.messages = [self.messages[0]]