"""scripts/eval.py — Agent 评测脚本

跑一批 QA 用例 + 一轮多轮对话测试，统计任务成功率、平均步数、平均耗时，
并验证记忆摘要压缩在真实 agent 循环中生效。

用法：
    python -m scripts.eval                    # 控制台报告
    python -m scripts.eval --markdown         # Markdown 报告（README 用）
    python -m scripts.eval --verbose          # 显示 agent 详细日志
    python -m scripts.eval --skip-multi-turn  # 跳过多轮测试（提速）
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass

from agent.loop import run_agent
from agent.models import AgentResult


# ============================================================
# QA 集：5 类场景，共 20 条
# ============================================================
@dataclass
class QAPair:
    question: str
    category: str
    expected_tools: list[str]      # 必须全部被调用才算通过
    expected_keywords: list[str]   # 空列表 = 不校验关键词；多个是 OR 语义


QA_PAIRS: list[QAPair] = [
    # ---------- 城市查询 (5) ----------
    QAPair("北京有哪些特点",           "城市查询", ["query_city_info"], ["北京"]),
    QAPair("介绍一下上海",             "城市查询", ["query_city_info"], ["上海"]),
    QAPair("广州的城市信息",           "城市查询", ["query_city_info"], ["广州"]),
    QAPair("深圳是一座怎样的城市",     "城市查询", ["query_city_info"], ["深圳"]),
    QAPair("杭州的城市信息",           "城市查询", ["query_city_info"], ["杭州"]),

    # # ---------- 天气查询 (5) ----------
    QAPair("北京天气怎么样",           "天气查询", ["get_weather"], ["北京"]),
    QAPair("上海今天天气如何",         "天气查询", ["get_weather"], ["上海"]),
    QAPair("广州会下雨吗",             "天气查询", ["get_weather"], ["广州"]),
    QAPair("深圳现在的气温",           "天气查询", ["get_weather"], ["深圳"]),
    QAPair("成都今天天气",             "天气查询", ["get_weather"], ["成都"]),

    # ---------- 用户关联 (4) ----------
    QAPair("用户1001所在城市的天气",   "用户关联",
           ["get_user_city", "get_weather"], []),
    QAPair("用户1002住在哪个城市",     "用户关联", ["get_user_city"], []),
    QAPair("查一下用户1003所在城市的天气", "用户关联",
           ["get_user_city", "get_weather"], []),
    QAPair("用户1001住在哪里",         "用户关联", ["get_user_city"], []),

    # ---------- 负向/抗幻觉 (1) ----------
    # 用户不存在时不应编造城市/天气，须诚实报告查不到
    QAPair("用户1004所在城市今天天气", "负向/抗幻觉",
           ["get_user_city"],
           ["未找到", "不存在", "未查询到", "无法确定", "未能"]),

    # ---------- RAG 检索 (5) ----------
    QAPair("文档里说 MAX_REPEAT 设成多少",       "RAG检索", ["search_docs"], ["3"]),
    QAPair("Function Calling 返回的是哪个字段",  "RAG检索", ["search_docs"], ["tool_calls"]),
    QAPair("ChromaDB 默认的英文嵌入模型叫什么",  "RAG检索", ["search_docs"], ["MiniLM"]),
    QAPair("ChromaDB 切换距离空间要怎么做",      "RAG检索", ["search_docs"],
           ["重建", "删除", "删", "重新写入", "新集合"]),
    # 负向：文档里确实没写这个阈值，模型应诚实说明未找到
    QAPair("工具返回超过多少字符会截断", "RAG检索", ["search_docs"],
       ["没有", "未找到", "不包含", "未提及", "未收录", "未给出", "无法给出"]),
]


# ============================================================
# 单轮 QA 核心逻辑
# ============================================================
def _check_success(result: AgentResult, qa: QAPair) -> tuple[bool, str]:
    """判断一次 agent 运行是否算成功。返回 (ok, reason)。"""
    if not result.success:
        return False, result.aborted_reason or "任务未完成"

    called = [r.name for r in result.tool_calls]
    called_set = set(called)

    missing = [t for t in qa.expected_tools if t not in called_set]
    if missing:
        return False, f"缺少工具调用 {missing}（实际: {called}）"

    if qa.expected_keywords:
        answer = result.answer or ""
        # OR 语义：命中任一关键词即可，避免同义表述（"未找到"/"未查询到"）漏判
        if not any(k in answer for k in qa.expected_keywords):
            return False, f"回答未命中任一关键词 {qa.expected_keywords}"

    return True, ""


def _run_case(qa: QAPair) -> dict:
    """跑单条用例，返回结果字典。"""
    t0 = time.perf_counter()
    try:
        result = run_agent(qa.question)   # session=None → 每次全新会话
        duration = time.perf_counter() - t0
        ok, reason = _check_success(result, qa)
        return {
            "qa": qa,
            "success": ok,
            "reason": reason,
            "steps": result.steps,
            "duration": duration,
            "tools": [r.name for r in result.tool_calls],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "qa": qa,
            "success": False,
            "reason": f"异常: {type(e).__name__}: {e}",
            "steps": 0,
            "duration": time.perf_counter() - t0,
            "tools": [],
        }


# ============================================================
# 多轮对话测试：验证记忆摘要压缩在真实 agent 循环中生效
# ============================================================
MULTI_TURN_QUESTIONS = [
    "北京天气怎么样",
    "那上海呢",
    "广州呢",
    "深圳现在的气温",
    "成都今天天气",
    "用户1001所在城市的天气",
    "用户1002住在哪个城市",
    "杭州的城市信息",
    "北京有哪些特点",
    "介绍一下上海",
]


def run_multi_turn_test() -> dict:
    """用一个共享 session 连续多轮提问，检查记忆压缩是否触发且上下文保持。

    返回：
        {
          "turns": int,
          "final_messages": int,
          "summary_chars": int,
          "compressed": bool,       # 是否至少压缩过一次
          "last_answer_ok": bool,   # 最后一轮回答是否成功
          "last_answer": str,
          "threshold": int,
        }
    """
    from agent.memory import AgentSession
    from agent.config import COMPRESS_THRESHOLD

    session = AgentSession()
    last_result = None

    for q in MULTI_TURN_QUESTIONS:
        last_result = run_agent(q, session=session)

    return {
        "turns": len(MULTI_TURN_QUESTIONS),
        "final_messages": len(session.messages),
        "summary_chars": len(session.summary or ""),
        "compressed": session.summary is not None,
        "last_answer_ok": bool(last_result and last_result.success),
        "last_answer": (last_result.answer if last_result else "") or "",
        "threshold": COMPRESS_THRESHOLD,
    }


# ============================================================
# 报告输出
# ============================================================
def _aggregate(cases: list[dict]) -> dict:
    total = len(cases)
    passed = sum(1 for c in cases if c["success"])
    rate = passed / total * 100 if total else 0.0

    success_cases = [c for c in cases if c["success"]]
    avg_steps = statistics.mean(c["steps"] for c in success_cases) if success_cases else 0.0
    avg_duration = statistics.mean(c["duration"] for c in cases) if cases else 0.0

    cats: dict[str, dict[str, int]] = {}
    for c in cases:
        cat = c["qa"].category
        cats.setdefault(cat, {"total": 0, "passed": 0})
        cats[cat]["total"] += 1
        if c["success"]:
            cats[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "rate": rate,
        "avg_steps": avg_steps,
        "avg_duration": avg_duration,
        "cats": cats,
        "failures": [c for c in cases if not c["success"]],
    }


def _print_console_report(cases: list[dict], multi_turn: dict | None = None) -> None:
    s = _aggregate(cases)
    line = "=" * 60

    print(line)
    print("Agent 评测报告")
    print(line)
    print(f"总用例数：{s['total']}")
    print(f"任务成功率：{s['passed']}/{s['total']} ({s['rate']:.1f}%)")
    print(f"平均步数：{s['avg_steps']:.2f}")
    print(f"平均耗时：{s['avg_duration']:.2f}s")
    print()

    print("按场景分类：")
    for cat, stat in s["cats"].items():
        pct = stat["passed"] / stat["total"] * 100 if stat["total"] else 0
        print(f"  {cat}: {stat['passed']}/{stat['total']} ({pct:.0f}%)")
    print()

    if s["failures"]:
        print("失败案例：")
        for i, c in enumerate(s["failures"], 1):
            print(f"  {i}. [{c['qa'].category}] Q: {c['qa'].question}")
            print(f"     → {c['reason']}")
            print(f"     → 实际调用工具: {c['tools']}")
        print()

    if multi_turn:
        print("多轮对话 / 记忆压缩：")
        mt = multi_turn
        mark = "✓" if mt["compressed"] and mt["last_answer_ok"] else "✗"
        print(f"  {mark} 轮数: {mt['turns']}  "
              f"压缩阈值: {mt['threshold']}  "
              f"最终消息数: {mt['final_messages']}  "
              f"摘要长度: {mt['summary_chars']} 字符")
        print(f"     最后一轮回答: {mt['last_answer'][:80]}"
              f"{'...' if len(mt['last_answer']) > 80 else ''}")


def _print_markdown_report(cases: list[dict], multi_turn: dict | None = None) -> None:
    s = _aggregate(cases)

    print("## Agent 评测结果\n")
    print("| 指标 | 数值 |")
    print("|---|---|")
    print(f"| 总用例数 | {s['total']} |")
    print(f"| 任务成功率 | {s['passed']}/{s['total']} ({s['rate']:.1f}%) |")
    print(f"| 平均步数 | {s['avg_steps']:.2f} |")
    print(f"| 平均耗时 | {s['avg_duration']:.2f}s |")
    print()
    print("| 场景 | 通过 / 总数 |")
    print("|---|---|")
    for cat, stat in s["cats"].items():
        print(f"| {cat} | {stat['passed']}/{stat['total']} |")

    if s["failures"]:
        print()
        print("<details><summary>失败案例</summary>\n")
        for i, c in enumerate(s["failures"], 1):
            print(f"{i}. **[{c['qa'].category}]** {c['qa'].question}")
            print(f"   - 原因：{c['reason']}")
            print(f"   - 实际调用工具：`{c['tools']}`")
        print("\n</details>")

    if multi_turn:
        mt = multi_turn
        ok = mt["compressed"] and mt["last_answer_ok"]
        print()
        print("### 多轮对话 / 记忆压缩\n")
        print("| 指标 | 数值 |")
        print("|---|---|")
        print(f"| 轮数 | {mt['turns']} |")
        print(f"| 压缩阈值（非 system 消息数） | {mt['threshold']} |")
        print(f"| 最终消息数 | {mt['final_messages']} |")
        print(f"| 是否触发压缩 | {'是' if mt['compressed'] else '否'} |")
        print(f"| 摘要长度 | {mt['summary_chars']} 字符 |")
        print(f"| 压缩后是否仍能正确回答 | {'是' if mt['last_answer_ok'] else '否'} |")
        print(f"| 结果 | {'✅' if ok else '❌'} |")


# ============================================================
# 入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 评测脚本")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 报告")
    parser.add_argument("--verbose", action="store_true", help="显示 agent 详细日志")
    parser.add_argument("--skip-multi-turn", action="store_true",
                        help="跳过多轮对话测试（省时间）")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("agent").setLevel(level)

    # ---------- 单轮 QA ----------
    total = len(QA_PAIRS)
    print(f"开始评测，共 {total} 条单轮用例...\n", file=sys.stderr)

    cases: list[dict] = []
    for i, qa in enumerate(QA_PAIRS, 1):
        print(f"[{i}/{total}] {qa.question}", file=sys.stderr, flush=True)
        case = _run_case(qa)
        mark = "✓" if case["success"] else "✗"
        print(
            f"        {mark}  steps={case['steps']}  {case['duration']:.2f}s",
            file=sys.stderr, flush=True,
        )
        cases.append(case)

    # ---------- 多轮对话 ----------
    multi_turn = None
    if not args.skip_multi_turn:
        print(
            f"\n开始多轮对话测试，共 {len(MULTI_TURN_QUESTIONS)} 轮...\n",
            file=sys.stderr,
        )
        multi_turn = run_multi_turn_test()
        mark = "✓" if multi_turn["compressed"] and multi_turn["last_answer_ok"] else "✗"
        print(
            f"        {mark}  messages={multi_turn['final_messages']}  "
            f"summary={multi_turn['summary_chars']}字符",
            file=sys.stderr, flush=True,
        )

    print(file=sys.stderr)

    if args.markdown:
        _print_markdown_report(cases, multi_turn)
    else:
        _print_console_report(cases, multi_turn)


if __name__ == "__main__":
    main()