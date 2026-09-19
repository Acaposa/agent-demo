# ============================================================
# app/api.py —— FastAPI 服务入口
# ------------------------------------------------------------
# 启动：uvicorn app.api:app --reload --port 8000
# API 模式：日志只写文件，控制台保持干净
# ============================================================

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent.loop import run_agent
from logging_config import setup_logging

setup_logging(console=False)
logger = logging.getLogger("api")

app = FastAPI(title="AI Agent API", version="1.0")


class ChatRequest(BaseModel):
    """聊天请求模型。"""
    model_config = ConfigDict(use_attribute_docstrings=True)

    question: str = Field(..., max_length=2000)
    """用户提出的问题文本"""


class ToolCallTrace(BaseModel):
    """单次工具调用的可观测信息。"""
    name: str
    args: dict
    result_len: int
    duration_ms: int
    error: str | None = None


class ChatResponse(BaseModel):
    """聊天响应模型。"""
    model_config = ConfigDict(use_attribute_docstrings=True)

    answer: str
    """Agent 生成的回答"""

    success: bool
    """是否成功完成任务"""

    steps: int = 0
    """Agent 执行的循环步数"""

    trace: list[ToolCallTrace] = []
    """工具调用轨迹"""

    aborted_reason: str | None = None
    """若失败，终止原因"""


@app.get("/health")
def health():
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """聊天接口：接收用户问题并返回 Agent 回答。"""
    logger.info(f"收到请求: {req.question}")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    result = run_agent(req.question)

    if not result.success:
        logger.warning(f"任务未能完成: {req.question} | {result.aborted_reason}")
        return ChatResponse(
            answer="抱歉，任务未能完成，请稍后重试。",
            success=False,
            steps=result.steps,
            trace=[ToolCallTrace(**r.__dict__) for r in result.tool_calls],
            aborted_reason=result.aborted_reason,
        )

    logger.info(f"请求完成: {req.question}")
    return ChatResponse(
        answer=result.answer or "",
        success=True,
        steps=result.steps,
        trace=[ToolCallTrace(**r.__dict__) for r in result.tool_calls],
    )