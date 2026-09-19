# AI Agent —— 基于 DeepSeek 的 Function Calling 智能助手

> **一句话定位**：一个从零手写的 AI Agent 框架，通过 Function Calling 驱动工具调用、RAG 知识库检索和多轮对话，完整复现了 LangChain 的核心能力，用于深入理解 Agent 架构设计。

---

## 技术亮点

- **手写 Function Calling 循环**：不依赖 LangChain，完整展示 ReAct 范式的 `observe → think → act` 机制
- **死循环反馈注入**：检测到重复调用时不直接终止，而是注入纠正消息让模型自我修正
- **中文 RAG 踩坑实践**：从 ChromaDB 默认英文模型到 `BAAI/bge-small-zh-v1.5`，检索距离从 0.65 降到 0.32
- **全链路可观测**：`ToolCallRecord` 记录每次工具调用的名称、参数、耗时、错误，API 层通过 `trace` 字段暴露
- **14 条 pytest 单测**：覆盖工具参数校验、循环终止、RAG 分块边界
- **双入口服务化**：CLI 交互 + FastAPI HTTP 接口，覆盖调试与部署场景

---

## 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                          用户入口                                 │
│          CLI (main.py)          FastAPI (app/api.py)             │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Agent 主循环 (agent/loop.py)                   │
│                                                                  │
│   ┌───────────┐    ┌──────────────┐    ┌────────────────────┐   │
│   │  LLM 层   │───▶│  工具调用层   │───▶│  结果回传 / 再推理  │   │
│   │ DeepSeek  │    │  tools.py    │    │  循环直到无 tool_calls│  │
│   └───────────┘    └──────────────┘    └────────────────────┘   │
│         ▲                  │                                     │
│         │                  ▼                                     │
│   ┌───────────┐    ┌──────────────────────────────────────┐     │
│   │ Memory 层 │    │           工具实现层                  │     │
│   │ memory.py │    │  get_user_city │ query_city_info      │     │
│   │ 滑动窗口   │    │  get_weather   │ search_docs (RAG)    │     │
│   └───────────┘    └──────────────────────────────────────┘     │
│                                     │                            │
└─────────────────────────────────────┼────────────────────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │  外部依赖             │
                          │  SQLite / wttr.in    │
                          │  ChromaDB + bge-zh   │
                          └──────────────────────┘
```

---

## 核心特性

- **手写 Agent 循环**：不依赖 LangChain，从零实现 `LLM → 工具调用 → 结果回传 → 再推理` 的完整循环，深入理解 Function Calling 机制
- **RAG 知识库检索**：基于 ChromaDB + 中文嵌入模型实现本地文档增量索引与语义检索，支持段落切分与 cosine 距离过滤
- **死循环防护（反馈注入）**：检测到重复工具调用时，不直接终止，而是将纠正反馈注入上下文，让模型自我修正
- **多轮对话记忆**：`AgentSession` 实现滑动窗口截断，保留最近 N 轮上下文
- **工具重试机制**：`execute_tool_with_retry` 对工具执行失败自动重试，提升稳定性
- **工具超长输出截断**：防止工具返回内容撑爆上下文窗口
- **任务超时保护**：单次任务设置 60 秒超时，防止无限等待
- **双入口支持**：CLI 交互式对话 + FastAPI HTTP 接口
- **完整可观测性**：`ToolCallRecord` 记录每次工具调用的名称、参数、耗时、错误信息

---

## 快速开始

**环境要求**：Python 3.10+

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
#    Linux / Mac:
cp .env.example .env
#    Windows:
copy .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxxx

# 3. 启动交互式对话
python main.py
```

> **首次运行提示**：RAG 模块会加载中文嵌入模型（约 100MB），首次使用需下载。
> 如网络不通，参考"已知限制"一节的本地化说明。

> 启动后直接输入问题即可开始对话，输入 `quit` / `exit` / `q` 退出。
> 单次问答：`python main.py "北京今天天气怎么样？"`

---

## API 服务

```bash
uvicorn app.api:app --reload --port 8000
```

调用：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "北京天气怎么样"}'
```

响应示例：

```json
{
  "answer": "北京目前 ☁️ 阴天，气温约 29°C。",
  "success": true,
  "steps": 2,
  "trace": [
    {
      "name": "get_weather",
      "args": {"city": "北京"},
      "result_len": 13,
      "duration_ms": 342,
      "error": null
    }
  ]
}
```

`trace` 字段记录每次工具调用的名称、参数、耗时和错误，可用于调试和监控。

---

## RAG 知识库

将 `.md` 文档放入 `docs/` 目录，Agent 会在下次检索时自动增量索引。特性：

- **段落切分**：`chunk_text` 按段落切分，超长段落按标点切，单句超长时硬切兜底
- **增量更新**：基于文件 MD5 哈希，只重建改动过的文档
- **cosine 距离过滤**：默认阈值 `0.5`，超过则视为不相关
- **中文嵌入**：`BAAI/bge-small-zh-v1.5`，约 100MB

手动全量重建索引：

```bash
python -m scripts.rebuild_index
```

---

## 可调参数

集中在 `agent/config.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `MAX_STEPS` | 10 | 单次任务最大循环步数 |
| `MAX_REPEAT` | 3 | 同一工具+参数允许重复次数（超出后注入反馈） |
| `MAX_TOOL_RESULT_CHARS` | 4000 | 工具输出截断阈值 |
| `MAX_QUESTION_CHARS` | 2000 | 用户问题最大长度（API 层校验） |
| `AGENT_TIMEOUT_SECONDS` | 60 | 单次任务总耗时上限 |

---

## 项目结构

```
.
├── main.py                 # CLI 入口（单次问答 + 交互式多轮）
├── requirements.txt        # 依赖清单
├── .env.example            # 环境变量示例（DEEPSEEK_API_KEY）
│
├── agent/                  # Agent 核心模块
│   ├── config.py           # 全局配置中心（路径、常量、System Prompt）
│   ├── llm_client.py       # LLM 客户端（延迟初始化）
│   ├── loop.py             # Agent 主循环
│   ├── memory.py           # 会话记忆（滑动窗口截断）
│   ├── models.py           # 数据结构（AgentResult / ToolCallRecord）
│   ├── rag.py              # RAG 检索（ChromaDB 增量索引 + 语义搜索）
│   └── tools.py            # 工具定义、参数校验、带重试执行
│
├── app/
│   └── api.py              # FastAPI 服务入口（/chat、/health）
│
├── scripts/
│   └── rebuild_index.py    # 全量重建 RAG 索引脚本
│
├── tests/                  # pytest 测试用例
│   ├── conftest.py         # 测试 fixtures（临时 DB / docs）
│   ├── test_loop.py        # Agent 循环逻辑、LLM 失败处理
│   ├── test_rag.py         # chunk_text 分块边界
│   └── test_tools.py       # 工具参数校验、工具执行
│
├── docs/                   # 知识库文档（Markdown，自动索引）
├── chroma_db/              # ChromaDB 持久化存储（自动生成）
├── logs/                   # 日志文件目录
└── cities.db               # SQLite 数据库（城市/用户数据）
```

---

## 设计决策

### 为什么手写循环而不用 LangChain？

LangChain 封装程度高，隐藏了大量底层细节。手写循环可以完整展示：

- `messages` 列表如何在每轮追加 assistant / tool 消息
- `tool_calls` 如何从 LLM 响应中解析并逐个执行
- 工具结果如何以 `role: tool` 回传并触发下一轮推理

这些是 Agent 的核心机制，手写一遍理解更深刻。实际工作中，是否用框架取决于团队和场景——手写适合理解原理和低依赖项目，LangGraph 适合有状态、多智能体的复杂流程。

### 为什么换中文嵌入模型？

ChromaDB 默认的嵌入模型是 `all-MiniLM-L6-v2`，只支持英文。项目文档以中文为主，实测中文查询的距离普遍在 0.6 以上，检索命中率低。换用 `BAAI/bge-small-zh-v1.5` 后，同一查询的距离降到 0.32，检索结果明显改善。

选型时也考虑了 `paraphrase-multilingual-MiniLM-L12-v2`，但 bge 在中文短文本检索上表现更好，模型也更小（约 100MB），适合本地部署。

### 死循环怎么防？

采用**反馈注入**策略，而非直接终止：

1. 记录每次工具调用的签名（`函数名:参数JSON`）
2. 同一签名重复**超过** `MAX_REPEAT`（默认 3 次，即第 4 次）时，不执行工具
3. 将纠正反馈（"你已重复调用 X 次，结果不会改变，请说明原因"）作为工具结果喂回模型
4. 模型收到反馈后会自我修正或直接告知用户

相比直接 `return`，这种方式给了模型纠错的机会。真正的无限循环由 `MAX_STEPS` 硬兜底。

### 为什么延迟初始化 LLM client？

`get_client()` 不在 import 时读取 `DEEPSEEK_API_KEY`，而是延迟到第一次调用。这样测试和 CI 环境没有 Key 也能正常 import 模块，不会因为一个环境变量缺失导致整套代码崩掉。

---

## 踩坑记录

1. **CMD 写中文文件乱码**：`echo 中文 > file.md` 默认 GBK 编码，Python 按 UTF-8 读会报错。改用 `Path.write_text(encoding="utf-8")` 或 VSCode 保存。
2. **嵌入模型选型**：ChromaDB 默认英文模型对中文语义几乎无效，换 `bge-small-zh-v1.5` 后距离从 0.65 降到 0.32。
3. **`chunk_text` 长文本边界**：无标点长文本（如 500 个 `a`）没有被硬切，会生成超长 chunk。单测暴露后加了 `while` 硬切兜底。
4. **`load_dotenv` 位置**：迁移目录结构后忘了搬，`os.environ` 读不到 Key。放到 `config.py` 顶部统一加载。
5. **模块级全局变量的 patch 陷阱**：`from config import DB_PATH` 是值拷贝，测试时改 `config.DB_PATH` 不影响 `tools.DB_PATH`，需要 patch 两处。
6. **ChromaDB 空间切换**：集合一旦创建，`hnsw:space` 不能修改。从默认 L2 切到 cosine 必须删除 `chroma_db/` 重建。

---

## 已知限制 / TODO

- [ ] **跨会话持久化记忆**：当前 `AgentSession` 仅在内存中，重启后丢失；计划接入 SQLite 或向量库实现长期记忆
- [ ] **记忆摘要压缩**：滑动窗口截断会丢失早期上下文，计划对超出窗口的历史做摘要压缩
- [ ] **嵌入模型本地化**：`sentence-transformers` 首次运行需从 HuggingFace 下载，可提供镜像指引或内置小型模型
- [ ] **流式输出**：当前 API 为同步返回，计划支持 SSE 流式响应
- [ ] **更多工具扩展**：计划增加网络搜索、代码执行等工具，验证框架的可扩展性
- [ ] **异步化**：当前工具执行是同步的，高并发场景下需要改为 async
- [ ] **评测体系**：目前没有固定 QA 集，无法量化 Agent 的任务成功率、工具调用准确率、平均步数

---

## 测试

```bash
# 运行全部测试
pytest -v

# 运行指定模块
pytest tests/test_tools.py -v
pytest tests/test_loop.py -v
pytest tests/test_rag.py -v
```

测试覆盖：

- `test_tools.py`：参数校验（未知工具 / 缺参 / 类型错误 / 正常）、城市查询、用户查询、非法城市天气
- `test_loop.py`：无工具调用时直接回答、LLM 调用失败时的错误处理
- `test_rag.py`：空文本、短段落、超长无标点文本硬切、多段落切分

---

## 后续计划

- [ ] 用 LangGraph 重写同样的 Agent，对比手写版和框架版的取舍
- [ ] 加一个垂直场景 Agent（代码助手 / 客服 / 数据分析），展示业务适配能力
- [ ] 补齐 Agent 评测：构造 30-50 条 QA 对，测任务成功率、工具调用准确率、平均步数、token 成本