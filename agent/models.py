# ============================================================
# agent/models.py —— 数据结构定义
# ============================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """单次工具调用的记录，用于可观测性和轨迹返回。"""
    name: str
    args: dict[str, Any]
    result_len: int
    duration_ms: int
    error: str | None = None


@dataclass
class AgentResult:
    """Agent 单次任务的完整结果。"""
    answer: str | None
    success: bool
    steps: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    aborted_reason: str | None = None