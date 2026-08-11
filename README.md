# MiniMate 🧠

> **工作/代码助手 Agent** —— 输入一个问题或任务，MiniMate 按你选择的模式执行：
> 纯问答（chat）一次调用直接回答，ReAct（react）带工具推理循环，Plan & Execute（plan）先规划再执行。

---

## 项目亮点

| 维度 | 说明 |
|------|------|
| **技术栈** | Python 3.12, DeepSeek API, Chroma RAG, FastAPI, Docker Compose |
| **核心能力** | 三种 Agent 模式、ReAct 推理循环、Plan & Execute、工具调用、记忆、RAG 检索 |
| **代码量** | 编排引擎、工具系统、记忆全部手写，零框架依赖 |

---

## 三种执行模式

| 模式 | 执行方式 | 适用场景 |
|------|---------|---------|
| `chat` | 单次 LLM 调用，直接返回答案 | 快速问答、概念解释 |
| `react` | Thought → Action → Observation → Final Answer 循环 | 需要工具计算/检索/搜索的任务 |
| `plan` | 先生成结构化 JSON 计划 → 逐步执行 → 汇总最终答案 | 复杂任务拆解、多步骤工作流 |

```
用户问题
   │
   ▼
Agent.run(mode)
   ├── chat:  LLM 一次调用 ────────────────▶ 答案
   ├── react: Thought → Action → Observation → Final Answer
   │              │ 工具调用（走 ToolExecutor）
   │              ▼
   │        web_search / calculator / save_file / query_knowledge
   └── plan: 生成 JSON 计划 → 逐步执行（每步走 react 或 chat）→ 汇总
```

### ReAct 循环

工具调用采用**双通道机制**：

- **Function Calling（默认）**：工具定义为 JSON Schema，模型 API 原生返回结构化 `tool_calls`，零正则解析、参数类型化、无歧义
- **文本协议（fallback）**：API 不支持 tools 时自动降级为 Thought/Action/Observation 文本 + 正则解析

```
Thought: 分析当前信息，决定下一步
Action: 选择工具（query_knowledge / save_file / ...）
Action Input: 传给工具的参数
Observation: 工具返回结果 → 回到 Thought（循环）

信息足够时输出：Final Answer: 最终回答
```

- **推理链累积**：每轮 Thought / Action / Observation 都追加进消息历史
- **完成判断**：模型自行输出 `Final Answer` 结束循环
- **防死循环**：`max_steps`（默认 8）兜底，超步数强制返回最后输出
- **Observation 截断**：工具返回过长时自动截断（默认 3000 字符）
- **容错解析**：文本协议兼容文本标记与 JSON 双格式
- **Schema 自动推断**：工具参数从函数签名自动生成 JSON Schema（inspect），无需手写

### Plan & Execute

1. **规划**：LLM 输出结构化 JSON 计划（2-5 步，含 step / goal / detail）
2. **执行**：逐步执行，前序结果作为下一步上下文（有工具走 ReAct，无工具走 chat）
3. **汇总**：把各步骤结果交给 LLM，生成最终完整答案

计划解析失败时自动回退到 ReAct，保证可用性。

---

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

### 2. 安装

```bash
uv sync
```

### 3. 使用

```bash
# 纯问答（单次 LLM 调用）
minimate "什么是 Python 装饰器" --mode chat

# ReAct 循环（可调工具）
minimate "计算 15*4 再减 3" --mode react

# Plan & Execute
minimate "对比 RAG 和微调方案" --mode plan

# 交互模式（多轮对话，短期记忆）
minimate

# 指定知识库目录
minimate "MCP 是什么" --kb-path ./docs --mode react

# 重建知识库索引
minimate --rebuild
```

### 交互模式

直接运行 `minimate`（不带参数）进入对话窗口：

- 打印 MiniMate ASCII Logo 与版本
- 会话记忆为**短期记忆**（内存中），多轮对话自动携带上下文
- `Ctrl+C` 退出，会话记忆随之清除

会话中可用命令：

```
/mode <chat|react|plan>   切换执行模式
/memory                   查看当前会话记忆
/clear                    清空会话记忆
/help                     显示帮助
/quit                     退出并清除记忆
```

也可以直接运行入口脚本：

```bash
python src/cli.py "你的问题" --mode plan
```

### 4. API 服务

```bash
minimate-server              # 安装后（等价于 python src/api.py）
# 或源码直跑
python src/api.py --port 8000
# 或 uvicorn 直连应用
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG", "mode": "react"}'
```

### 5. Docker 部署

```bash
docker compose build
docker compose up -d
docker compose exec minimate minimate --rebuild  # 首次重建知识库
docker compose logs -f
```

### 6. MCP 接入（可选）

项目支持**本地工具 + MCP 远程工具**双工具源，MCP 支持两种传输：

- `stdio`：本地子进程连接（通过 stdin/stdout 与 MCP Server 进程通信）
- `http`：远程 URL 连接（Streamable HTTP，支持认证 headers）

复制配置模板并填写 MCP Server：

```bash
cp config.example.json config.json
```

```json
{
  "mcp": {
    "servers": [
      {
        "name": "demo",
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "python", "examples/mcp_demo_server.py"],
        "env": {}
      },
      {
        "name": "notion",
        "transport": "http",
        "url": "https://mcp.notion.com/mcp",
        "headers": {}
      }
    ]
  }
}
```

启动时自动连接配置的 MCP Server，将其工具（`tools/list`）包装进工具系统，Agent 的 Function Calling 循环无感知。

交互模式下可用 `/mcp` 查看各服务器连接状态：

```
> /mcp
  MCP 服务器状态：
    ✅ demo [stdio] connected · 3 个工具
    ✅ notion [http] connected · N 个工具
```

示例 Server：`examples/mcp_demo_server.py`（stdio，add/multiply/current_time）、`examples/mcp_http_server.py`（http，greeting/reverse）。

---

## 项目结构

```
MiniMate/
├── src/
│   ├── cli.py                 # CLI 入口（三模式）
│   ├── api.py                 # FastAPI 服务（三模式）
│   └── minimate/
│       ├── __init__.py           # 包定义
│       ├── orchestrator.py       # Agent / Task / Crew + 三模式执行
│       ├── tools/                # 工具系统包（按类分组注册）
│       │   ├── core.py           # Tool / ToolExecutor / ReAct 解析
│       │   ├── file_tools.py     # 文件工具（register_file_tools）
│       │   ├── shell_tools.py    # Shell 命令工具（register_shell_tools）
│       │   ├── web_tools.py      # 搜索工具（register_web_tools）
│       │   ├── rag_tools.py      # 知识库工具（register_rag_tools）
│       │   └── registry.py       # 注册中心（register_all_tools）
│       ├── memory.py             # 三层记忆（Buffer / Entity / Findings）
│       ├── llm.py                # LLM API 封装
│       ├── config.py             # 配置模块（config.json 加载）
│       ├── mcp/                  # MCP 包（stdio + http 双传输）
│       │   ├── adapter.py        # McpToolAdapter：统一适配器 + 连接状态机
│       │   ├── stdio.py          # stdio 传输（本地子进程）
│       │   ├── http.py           # Streamable HTTP 传输（远程 URL）
│       │   └── oauth.py          # OAuth 2.0（授权码 PKCE / 设备流 / 动态注册）
│       └── rag/                  # RAG 知识库（Chroma + bge）
│           ├── __init__.py
│           ├── config.toml       # 模型路径、目录配置
│           ├── knowledge.py      # 文档加载 + Chroma 检索
│           └── repo/             # 知识库文档（.md/.txt，用户自放）
├── tests/
│   └── test_react.py             # 单元测试（mock，无需 API）
├── examples/
│   └── mcp_demo_server.py        # 示例 MCP Server（FastMCP）
│   └── mcp_http_server.py        # 示例远程 MCP Server（FastMCP HTTP）
├── docs/                         # 设计文档
├── output/                       # 工具输出目录
├── config.example.json           # 配置模板（MCP servers）
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 设计理念

### 为什么手写编排引擎

| 方案 | 优点 | 缺点 |
|------|------|------|
| CrewAI | 开箱即用 | 黑盒、难排查、面试讲不清原理 |
| LangGraph | 状态机灵活 | 学习成本高、抽象层厚 |
| **手写** | 完全可控、零依赖 | 需要造轮子 |

### Agent / Task / Crew

| 概念 | 职责 |
|------|------|
| Agent | 角色 + 工具 + 记忆 + 执行模式（chat / react / plan） |
| Task | 任务单元：描述 + 绑定 Agent + 模式 |
| Crew | 任务调度器：顺序执行 + 上下文传递 |

### 记忆系统

| 层次 | 作用 |
|------|------|
| Buffer | 短期对话缓存，CircularBuffer 自动丢弃 |
| Entity | 实体提取（人名/术语/概念） |
| Findings | 关键发现持久化，跨任务传递 |

---

## 路线图

- [x] 三种执行模式（chat / react / plan）
- [x] ReAct 推理-行动循环（Thought / Action / Observation + 完成判断）
- [x] Plan & Execute（结构化计划 + 逐步执行 + 汇总）
- [x] 工具调用系统 + 按需装配
- [x] Function Calling 工具调用（双通道：FC 优先 + 文本协议 fallback）
- [x] MCP 协议化工具网关（本地工具 + MCP 远程工具共存）
- [x] 代码助手工具（read_file / write_file / list_files / find_files / grep_files / run_shell）
- [x] RAG 知识库（Chroma + bge 本地模型）
- [ ] Skill 技能包系统
- [ ] 长期记忆持久化 + 向量检索

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12 | 运行环境 |
| DeepSeek API | LLM 推理 |
| Chroma + bge-small-zh | RAG 知识库向量检索 |
| FastAPI + Uvicorn | REST API 服务 |
| Docker / Compose | 容器化部署 |
| 零框架依赖 | 编排引擎、工具、记忆全部手写 |
