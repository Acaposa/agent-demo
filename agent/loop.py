# ============================================================
# agent/loop.py —— Agent 主循环
# ------------------------------------------------------------
# 一次 run_agent 处理一个用户任务，返回 AgentResult。
# 循环逻辑：LLM → 工具调用 → 结果回传 → 再推理，直到无 tool_calls。
# ============================================================

import json
import logging
import time

from agent.config import MAX_STEPS, MAX_REPEAT, MAX_TOOL_RESULT_CHARS, AGENT_TIMEOUT_SECONDS
from agent.llm_client import get_client
from agent.memory import AgentSession
from agent.models import AgentResult, ToolCallRecord
from agent.tools import tools, validate_args, execute_tool_with_retry

logger = logging.getLogger("agent")


def _truncate_tool_result(result: str) -> str:
    """工具输出超长时截断，避免撑爆上下文。"""
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + f"\n...[已截断，原始长度 {len(result)} 字符]"
    )


def _is_tool_error(result: str) -> bool:
    """判断工具返回值是否代表执行失败，用于写入 ToolCallRecord.error。"""
    if result.startswith("未知工具："):
        return True
    if "执行失败（已重试" in result:
        return True
    return False


def run_agent(user_input: str, session: AgentSession | None = None) -> AgentResult:
    """
    处理一次用户任务。

    参数：
        user_input: 用户问题
        session: 可选的会话对象，传入时可实现多轮对话

    返回：
        AgentResult，包含 answer / success / steps / tool_calls
    """
    if session is None:
        session = AgentSession()
    session.add_user(user_input)
    start_time = time.perf_counter()
    tool_call_history: list[str] = []
    records: list[ToolCallRecord] = []

    try:
        for step in range(1, MAX_STEPS + 1):
            # 总耗时超限则提前退出
            elapsed = time.perf_counter() - start_time
            if elapsed > AGENT_TIMEOUT_SECONDS:
                logger.warning(f"任务超时（{elapsed:.1f}s > {AGENT_TIMEOUT_SECONDS}s），终止。")
                return AgentResult(
                    answer=None,
                    success=False,
                    steps=step - 1,
                    tool_calls=records,
                    aborted_reason=f"任务超时（{elapsed:.1f}s）"
                )
            logger.info(f"--- Step {step} ---")

            # ---------- 调用 LLM ----------
            try:
                resp = get_client().chat.completions.create(
                    model="deepseek-chat",
                    messages=session.messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"调用模型失败：{e}")
                return AgentResult(
                    answer=None,
                    success=False,
                    steps=step,
                    tool_calls=records,
                    aborted_reason=f"LLM 调用失败：{e}"
                )

            msg = resp.choices[0].message
            raw_content = msg.content or ""

            # ---------- 记录 LLM 返回结果 ----------
            if msg.tool_calls:
                logger.info(
                    f"LLM 返回 {len(msg.tool_calls)} 个工具调用: "
                    f"{[tc.function.name for tc in msg.tool_calls]}"
                )
            else:
                logger.info(
                    f"LLM 返回文本: {raw_content[:200]}"
                    f"{'...' if len(raw_content) > 200 else ''}"
                )

            # ---------- 把 assistant 消息写回 session ----------
            assistant_tool_calls = None
            if msg.tool_calls:
                assistant_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in msg.tool_calls
                ]
            session.add_assistant(msg.content, assistant_tool_calls)

            # ---------- 没有工具调用，收尾 ----------
            if not msg.tool_calls:
                logger.info(f"最终回答: {msg.content}")
                return AgentResult(
                    answer=msg.content or "",
                    success=True,
                    steps=step,
                    tool_calls=records
                )

            # ---------- 逐个执行工具调用 ----------
            for tc in msg.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments

                # 参数解析
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    result = f"参数解析失败：{e}。请检查参数格式。"
                    logger.warning(result)
                    session.add_tool_result(tc.id, _truncate_tool_result(result))
                    records.append(ToolCallRecord(
                        name=name, args={}, result_len=len(result),
                        duration_ms=0, error="json_decode"
                    ))
                    continue

                # 参数校验
                ok, err = validate_args(name, args)
                if not ok:
                    result = f"参数校验失败：{err}"
                    logger.warning(result)
                    session.add_tool_result(tc.id, _truncate_tool_result(result))
                    records.append(ToolCallRecord(
                        name=name, args=args, result_len=len(result),
                        duration_ms=0, error=err
                    ))
                    continue

                # 死循环检测：重复超过阈值时注入反馈，让模型自我纠正
                signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                tool_call_history.append(signature)
                repeat_count = tool_call_history.count(signature)

                if repeat_count > MAX_REPEAT:
                    feedback = (
                        f"你已经用完全相同的参数调用 {name} 超过 {MAX_REPEAT} 次，"
                        f"结果不会改变。请不要再用相同参数调用该工具。"
                        f"如果无法完成，请直接向用户说明原因。"
                    )
                    logger.warning(
                        f"重复调用 {signature} {repeat_count} 次，注入纠正反馈"
                    )
                    session.add_tool_result(tc.id, feedback)
                    records.append(ToolCallRecord(
                        name=name, args=args, result_len=len(feedback),
                        duration_ms=0, error="repeated_call"
                    ))
                    continue  # 不执行工具，把反馈喂回模型

                # 执行工具
                logger.info(f"执行工具 {name}({args})")
                t0 = time.perf_counter()
                result = execute_tool_with_retry(name, args)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(f"工具返回 {len(result)} 字符: {result[:120]}")

                # 工具本身失败时把错误信息也记进 trace
                tool_error = result if _is_tool_error(result) else None

                session.add_tool_result(tc.id, _truncate_tool_result(result))
                records.append(ToolCallRecord(
                    name=name, args=args, result_len=len(result),
                    duration_ms=duration_ms, error=tool_error
                ))

        # 达到最大步数
        logger.warning(f"达到最大循环次数 {MAX_STEPS}，Agent 未能完成任务。")
        return AgentResult(
            answer=None,
            success=False,
            steps=MAX_STEPS,
            tool_calls=records,
            aborted_reason=f"达到最大循环次数 {MAX_STEPS}"
        )

    finally:
        session.save()