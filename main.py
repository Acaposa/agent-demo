# ============================================================
# main.py —— CLI 入口
# ------------------------------------------------------------
# 单次问答：python main.py 你的问题
# 交互式：python main.py
# 带会话恢复：python main.py --session <id>
# ============================================================

import argparse

from agent.loop import run_agent
from agent.memory import AgentSession
from logging_config import setup_logging


def main():
    setup_logging(console=True)

    parser = argparse.ArgumentParser(description="AI Agent CLI")
    parser.add_argument("question", nargs="?", help="单次提问，留空进入交互式")
    parser.add_argument("--session", help="会话 ID，用于恢复历史（默认不持久化）")
    args = parser.parse_args()

    # 有 --session 就加载/创建持久化 session，否则用普通内存 session
    if args.session:
        session = AgentSession.load_or_create(args.session)
        print(f"[会话 {args.session}] 已加载，{len(session.messages)} 条历史消息")
    else:
        session = AgentSession()

    # ---------- 单次问答 ----------
    if args.question:
        print(f"你：{args.question}")
        result = run_agent(args.question, session=session)
        if result.success:
            print(f"Agent：{result.answer}")
        else:
            print(f"[失败] {result.aborted_reason or '任务未完成'}")
        return

    # ---------- 交互式多轮 ----------
    print("=== AI Agent 已启动 ===")
    print("输入问题开始对话，输入 quit / exit / q 退出\n")

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