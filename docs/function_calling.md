# Function Calling 原理

Function Calling 是 LLM 在生成时返回结构化工具调用意图的能力。
DeepSeek 的 function calling 兼容 OpenAI 接口，请求体需带 `tools` 参数，
`tool_choice` 可选 `auto` / `none` / 指定函数名。

一次完整的工具调用流程：

1. 客户端把工具 JSON Schema 放进 `tools` 字段
2. LLM 判断需要调用工具，返回 `message.tool_calls`
3. 客户端执行工具，把结果以 `role: tool` 消息回传
4. LLM 基于工具结果生成最终回答

本项目设置 `MAX_REPEAT=3`：同一工具用相同参数调用超过 3 次，
注入纠正消息而非直接终止；`MAX_STEPS=10` 作为兜底。

注意：工具参数由 LLM 生成，可能是非法 JSON，必须 try/except json.JSONDecodeError。