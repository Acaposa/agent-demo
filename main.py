# ============================================================
# main.py —— CLI 入口
# ------------------------------------------------------------
# 命令行单次问答：python main.py 你的问题
# 交互式多轮对话：直接运行，quit/exit/q 退出
# ============================================================

import sys

from agent.loop import run_agent
from agent.memory import AgentSession
from logging_config import setup_logging


def main():
    setup_logging(console=True)

    # ---------- 单次问答 ----------
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"你：{question}")
        result = run_agent(question)
        if result.success:
            print(f"Agent：{result.answer}")
        else:
            print(f"[失败] {result.aborted_reason or '任务未完成'}")
        return

    # ---------- 交互式多轮 ----------
    print("=== AI Agent 已启动 ===")
    print("输入问题开始对话，输入 quit / exit / q 退出\n")

    session = AgentSession()

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = run_agent(user_input, session=session)
        if result.success:
            print(f"Agent：{result.answer}")
        else:
            print(f"[失败] {result.aborted_reason or '任务未完成'}")


if __name__ == "__main__":
    main()