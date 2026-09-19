# ============================================================
# agent/tools.py —— 工具定义、参数校验、带重试执行
# ------------------------------------------------------------
# 从原 agent_demo.py 抽出的工具层。零依赖主循环，可独立测试。
# ============================================================

import json
import logging
import sqlite3
import time
from contextlib import closing

import requests

from agent.config import DB_PATH
from agent.rag import search_docs

logger = logging.getLogger("tools")


# ------------------------------------------------------------
# 数据库初始化
# ------------------------------------------------------------
def init_db():
    """初始化 SQLite 数据库（cities / users 两张表）。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS cities (
                name TEXT PRIMARY KEY,
                province TEXT,
                population INTEGER
            )
        """)
        c.execute("INSERT OR REPLACE INTO cities VALUES ('北京', '北京市', 2189)")
        c.execute("INSERT OR REPLACE INTO cities VALUES ('上海', '上海市', 2487)")
        c.execute("INSERT OR REPLACE INTO cities VALUES ('广州', '广东省', 1881)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                city TEXT
            )
        """)
        c.execute("INSERT OR REPLACE INTO users VALUES ('1001', '张三', '北京')")
        c.execute("INSERT OR REPLACE INTO users VALUES ('1002', '李四', '上海')")
        c.execute("INSERT OR REPLACE INTO users VALUES ('1003', '王五', '广州')")

        conn.commit()


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def query_city_info(city: str) -> str:
    """根据城市名查询省份和人口。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT name, province, population FROM cities WHERE name = ?",
            (city,)
        ).fetchone()

    if row:
        return f"{row[0]}位于{row[1]}，人口约{row[2]}万"
    return f"数据库中没有找到{city}的信息"


INVALID_CITIES = {
    "火星", "月球", "太阳", "木星", "土星", "金星", "水星", "地球",
    "冥王星", "海王星", "天王星", "比邻星", "半人马座", "银河系"
}


def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    if city in INVALID_CITIES:
        return f"查询天气失败：{city}不是地球城市，无法查询天气。请提供地球上的城市名。"

    try:
        r = requests.get(f"https://wttr.in/{city}?format=3&lang=zh", timeout=10)
        r.raise_for_status()
        return r.text.strip()
    except Exception as e:
        return f"查询天气失败：{e}"


def get_user_city(user_id: str) -> str:
    """根据用户 ID 查询姓名和所在城市。"""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT name, city FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

    if row:
        return f"用户{row[0]}(ID: {user_id}）所在城市是{row[1]}"
    return f"未找到ID为{user_id}的用户"


# ------------------------------------------------------------
# 工具 schema（给 LLM 的 function calling 定义）
# ------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": "根据用户ID查询该用户所在的城市。当你不知道用户所在城市时，必须先调用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID，例如 1001"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "检索本地知识库文档。\n"
                "当用户询问任何可能记录在项目文档中的内容时使用，包括但不限于 Agent 开发、RAG、Function Calling、Python 编程技巧、项目配置等。\n"
                "优先查文档，不要凭训练时的记忆直接回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_city_info",
            "description": "查询数据库中某个城市的基本信息，包括省份和人口，如果用户问的是非地球城市，提示错误",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如 北京"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如 北京"}
                },
                "required": ["city"]
            }
        }
    }
]


# ------------------------------------------------------------
# 参数校验
# ------------------------------------------------------------
TOOL_SCHEMAS = {
    "get_user_city": {"required": ["user_id"], "types": {"user_id": str}},
    "query_city_info": {"required": ["city"], "types": {"city": str}},
    "get_weather": {"required": ["city"], "types": {"city": str}},
    "search_docs": {"required": ["query"], "types": {"query": str}},
}


def validate_args(name: str, args: dict) -> tuple[bool, str]:
    """校验工具名称与参数（存在性 / 必填 / 类型）。"""
    schema = TOOL_SCHEMAS.get(name)
    if not schema:
        return False, f"未知工具：{name}"

    for req in schema["required"]:
        if req not in args:
            return False, f"缺少必要参数：{req}"
        if not isinstance(args[req], schema["types"][req]):
            return False, f"参数 {req} 类型错误，期望 {schema['types'][req].__name__}"

    return True, ""


# ------------------------------------------------------------
# 工具注册表：name → 执行函数
# 加新工具时，只要在 tools 列表里写 schema，在这里注册函数即可
# ------------------------------------------------------------
TOOL_REGISTRY = {
    "get_user_city": get_user_city,
    "query_city_info": query_city_info,
    "get_weather": get_weather,
    "search_docs": search_docs,
}


def execute_tool_with_retry(name: str, args: dict, max_retries: int = 2) -> str:
    """执行工具，失败时自动重试 max_retries 次。"""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"未知工具：{name}"

    for attempt in range(max_retries + 1):
        try:
            return func(**args)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return f"工具 {name} 执行失败（已重试 {max_retries} 次）：{e}"