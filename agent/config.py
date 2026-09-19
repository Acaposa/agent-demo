# ============================================================
# agent/config.py —— 全局配置中心
# ------------------------------------------------------------
# 所有路径、常量、System Prompt 都从这里取。
# 其他模块禁止再用 Path(__file__).parent 自己算路径。
# ============================================================

from pathlib import Path
from dotenv import load_dotenv
# ---------- 路径 ----------
# __file__ = <project>/agent/config.py
# .resolve().parent       -> <project>/agent
# .resolve().parent.parent-> <project>
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env（在任何 os.environ 读取之前）
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
DB_PATH = BASE_DIR / "cities.db"
LOG_DIR = BASE_DIR / "logs"

# ---------- LLM ----------
SYSTEM_PROMPT = """你是一个严谨的助手。
调用工具失败时，根据错误信息决定是否重试或告知用户。
最终回答必须基于工具返回的真实数据。
不要替工具判断城市是否有效，把判断交给工具执行结果。
如果工具返回的数据明显不合理（例如火星天气、不存在的城市、与常识严重冲突），不要在回答中呈现该数据的具体数值，直接说明查询失败或数据不可靠，并建议用户换一个查询。
当用户询问技术概念、编程知识、项目文档相关内容时，必须先调用 search_docs 检索，基于检索结果回答。只有检索不到时才用通用知识补充，并明确标注"以下内容来自通用知识，非项目文档"。"""

# ---------- Agent 循环控制 ----------
MAX_STEPS = 10                   # 单次任务最大循环步数
MAX_REPEAT = 3                   # 同一工具+参数允许重复次数（P0 会用）

# ---------- 输入/输出约束（P3 会用，先占位）----------
MAX_TOOL_RESULT_CHARS =  4000  # 工具返回内容最大字符数
MAX_QUESTION_CHARS = 2000        # 用户问题最大字符数
AGENT_TIMEOUT_SECONDS = 60       # 单次任务总耗时上限