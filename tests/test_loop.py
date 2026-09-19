# ============================================================
# tests/test_loop.py —— 主循环单测（不调真实 LLM）
# ============================================================

from unittest.mock import MagicMock, patch

from agent.loop import run_agent
from agent.models import AgentResult


def _mock_response(content=None, tool_calls=None):
    """构造一个 mock 的 LLM 响应。"""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp

def _mock_tool_call(tc_id: str, name: str, args_json: str):
    """构造一个 mock 的 tool_call 对象。"""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = args_json
    return tc

class TestRunAgent:
    def test_direct_answer_no_tool(self):
        """LLM 直接回答，不调工具。"""
        with patch("agent.loop.get_client") as mock_get:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response(
                content="你好，我是 Agent。"
            )
            mock_get.return_value = client

            result = run_agent("你好")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.answer == "你好，我是 Agent。"
        assert result.steps == 1
        assert result.tool_calls == []

    def test_llm_failure_returns_unsuccessful(self):
        """LLM 调用异常时返回 success=False。"""
        with patch("agent.loop.get_client") as mock_get:
            client = MagicMock()
            client.chat.completions.create.side_effect = RuntimeError("network down")
            mock_get.return_value = client

            result = run_agent("测试")

        assert result.success is False
        assert result.answer is None
        assert "LLM 调用失败" in (result.aborted_reason or "")


    def test_timeout_returns_unsuccessful(self):
        """任务超时时返回 success=False，且 aborted_reason 含'超时'。"""
        with patch("agent.loop.get_client") as mock_get, \
             patch("agent.loop.AGENT_TIMEOUT_SECONDS", -1):
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response(
                content="不会走到这里"
            )
            mock_get.return_value = client

            result = run_agent("测试超时")

        assert result.success is False
        assert result.answer is None
        assert "超时" in (result.aborted_reason or "")

    def test_repeated_tool_call_injects_feedback(self):
        """
        重复调用同一工具时注入 feedback（error='repeated_call'），
        最终由 MAX_STEPS 兜底返回失败。
        """
        tc = _mock_tool_call("call_1", "get_weather", '{"city": "北京"}')

        with patch("agent.loop.get_client") as mock_get, \
             patch("agent.loop.MAX_STEPS", 5), \
             patch("agent.loop.MAX_REPEAT", 2), \
             patch("agent.loop.execute_tool_with_retry", return_value="北京: 晴 25°C"):
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response(
                content=None, tool_calls=[tc]
            )
            mock_get.return_value = client

            result = run_agent("北京天气")

        # MAX_STEPS 兜底失败
        assert result.success is False
        assert "最大循环次数" in (result.aborted_reason or "")

        # 至少有 3 条 repeated_call 记录（第 3、4、5 步触发反馈注入）
        errors = [r.error for r in result.tool_calls]
        assert errors.count("repeated_call") >= 3

        # 前 2 次是正常执行，无 error
        assert errors.count(None) == 2    