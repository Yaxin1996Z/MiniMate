# MiniMate 🧠

> **工作/代码助手 Agent** —— 输入一个问题或任务，MiniMate 按你选择的模式执行：
> 纯问答（chat）一次调用直接回答，ReAct（react）带工具推理循环，Plan & Execute（plan）先规划再执行，
> Multi-Agent（multi）规划者拆解、执行者并行干活、检查者验收的三角色协作。

---

## 项目亮点

| 维度 | 说明 |
|------|------|
| **技术栈** | Python 3.12, DeepSeek API, Chroma RAG, FastAPI, Docker Compose |
| **核心能力** | 四种 Agent 模式（chat / react / plan / multi）、ReAct 推理循环、Multi-Agent 编排、工具调用、三层记忆、RAG 检索 |
| **代码量** | 编排引擎、工具系统、记忆、多 Agent 协作全部手写，零框架依赖 |

---

## 四种执行模式

| 模式 | 执行方式 | 适用场景 |
|------|---------|---------|
| `chat` | 单次 LLM 调用，直接返回答案 | 快速问答、概念解释 |
| `react` | Thought → Action → Observation → Final Answer 循环 | 需要工具计算/检索/搜索的任务 |
| `plan` | 先生成结构化 JSON 计划 → 逐步执行 → 汇总最终答案 | 复杂任务拆解、多步骤工作流 |
| `multi` | 规划者拆解 → 多 Worker 并行执行 → 检查者逐步骤验收 → 汇总 | 复杂任务且对结果质量有要求 |

```
用户问题
   │
   ▼
Agent.run(mode)
   ├── chat:  LLM 一次调用 ────────────────▶ 答案
   ├── react: Thought → Action → Observation → Final Answer
   │              │ 工具调用（走 ToolExecutor）
   │              ▼
   │        read_file / write_file / run_shell / web_search / search_code
   └── plan: 生成 JSON 计划 → 逐步执行（每步走 react 或 chat）→ 汇总

   multi: 规划者拆步骤 → 执行者并行干活 → 检查者验收 → 汇总
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

### Multi-Agent 协作

把全能型 Agent 拆成三个专职角色，像项目团队一样分工协作、互相制衡。采用**主从架构（Orchestrator-SubAgent）**：编排器是"主"，负责任务分发与流程控制；子 Agent 是"从"，彼此不直接对话，所有消息都经编排器路由。

| 角色 | 职责 | 是否调用工具 |
|------|------|-------------|
| **规划者（Planner）** | 把用户需求拆成 JSON 执行计划（id / 描述 / 类型 / 依赖） | 否 |
| **执行者（Worker）** | 按步骤调用工具完成具体操作（唯一可调工具的角色） | 是 |
| **检查者（Reviewer）** | 验收执行结果，通过放行 / 不通过打回重做 | 否 |

**六阶段工作流：**

```
用户任务
  │
  ▼
① 规划   规划者输出 JSON 计划
② 解析   建模为 ExecutionStep，建立依赖 DAG（拓扑排序分层）
③ 执行   同层无依赖步骤多 Worker 并行执行（线程池 + 轮询分配）
④ 审查   检查者逐步骤验收，不通过注入问题清单重试（最多 2 次）
⑤ 残留   依赖失败的步骤显式标记 SKIPPED，提示用户
⑥ 汇总   按步骤顺序汇总结果，生成最终答案
```

关键设计：

- **保守审批策略**：检查者输出无法解析（空内容 / 非 JSON / 缺 approved 字段）时默认"不通过"，宁可多审一次，不放过一个问题
- **反馈重试闭环**：审查不通过时，把问题列表与改进建议注入执行者上下文重做，最多 2 次，超过后保留当前结果并提示人工复核
- **依赖上下文注入**：执行某步骤前注入已完成依赖步骤的结果摘要（截断 500 字符），Worker 知道前一步干了什么又不会撑爆上下文
- **每步干净状态**：每个子 Agent 独立对话历史，步骤完成后清空（保留系统提示词），避免上一步干扰下一步判断

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

# Multi-Agent 协作（规划者/执行者/检查者）
minimate "重构工具注册模块" --mode multi

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
/mode <chat|react|plan|multi>   切换执行模式
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

## Agent 评测

内置可复现的评测体系，用于量化 Agent 在四种模式下的任务完成能力，面试与简历数据均出自真实运行报告。

```bash
# 运行全部评测（16 条用例，覆盖 chat / react / plan / multi）
minimate --eval basic

# 只评测指定模式（快速验证）
minimate --eval basic --eval-modes react,multi

# 只跑前 N 条（验证链路用）
minimate --eval basic --eval-max 2
```

运行后生成报告（Markdown + JSON）到 `eval/results/<时间戳>/`，逐条展示通过/失败、判定理由、耗时与 Token 消耗。

**设计要点（保证评测可信）：**

- **确定性断言 checker**：判定基于文件存在 / 内容包含 / 数值比对 / 命令退出码，不使用 LLM 自评，结果可复现
- **沙箱隔离**：每条用例在独立临时目录执行，工具只能操作沙箱内文件，互不影响
- **本地确定性任务**：评测集不依赖网络，覆盖四模式（问答 / 工具调用 / 计划执行 / 多 Agent 协作）
- **成本可观测**：报告记录每条用例的耗时与 Token 消耗，便于评估优化效果

---

## 项目结构

```
MiniMate/
├── src/
│   ├── cli.py                 # CLI 入口（四模式）
│   ├── api.py                 # FastAPI 服务
│   └── minimate/
│       ├── __init__.py           # 包定义
│       ├── orchestrator.py       # 兼容入口（re-export Agent / 计划 / 多 Agent）
│       ├── agent/                # Agent 包（单 Agent + 多 Agent）
│       │   ├── agent.py          # 单 Agent：chat / react / plan 三模式
│       │   ├── task.py           # PlanTask + 计划解析 + DAG 拓扑排序
│       │   ├── history_compactor.py  # LLM 消息历史压缩
│       │   └── multi/            # 多 Agent 编排（主从架构 + 三角色）
│       │       ├── role.py       # AgentRole 枚举（规划者/执行者/检查者）
│       │       ├── message.py    # AgentMessage 六种消息类型
│       │       ├── sub_agent.py  # SubAgent：独立角色 + 共享工具/记忆
│       │       └── orchestrator.py  # 编排器：规划→执行→审查→汇总
│       ├── tools/                # 工具系统包（按类分组注册）
│       │   ├── core.py           # Tool / ToolExecutor / ReAct 解析
│       │   ├── file_tools.py     # 文件工具（register_file_tools）
│       │   ├── shell_tools.py    # Shell 命令工具（register_shell_tools）
│       │   ├── web_tools.py      # 搜索工具（register_web_tools）
│       │   ├── rag_tools.py      # 知识库工具（register_rag_tools）
│       │   └── registry.py       # 注册中心（register_all_tools）
│       ├── memory/               # 三层记忆（短期 / 长期 / 工具）
│       │   ├── manager.py        # MemoryManager：四类记忆整合 + 上下文组装
│       │   ├── short_term.py     # 短期：Token 预算淘汰 + Map-Reduce 压缩
│       │   ├── long_term.py      # 长期：SQLite 持久化 + 去重 + 作用域
│       │   ├── tool_memory.py    # 工具记忆：循环内重复调用检测
│       │   ├── compressor.py     # Map-Reduce 压缩器 + LLM 事实提取
│       │   └── token_budget.py   # Token 预算控制
│       ├── llm/                  # LLM 统一调用（client / chat / stats）
│       ├── mcp/                  # MCP 包（stdio + http 双传输 + OAuth）
│       ├── coderag/              # 代码仓库 RAG（AST 索引 + BM25 + bge 向量）
│       └── rag/                  # 文档知识库（Chroma + bge）
├── tests/
│   ├── test_cli.py               # CLI / 交互模式测试
│   ├── test_multi.py             # 多 Agent 编排测试
│   ├── test_react.py             # ReAct 循环测试
│   └── test_memory.py            # 记忆系统测试
├── examples/
│   └── mcp_demo_server.py        # 示例 MCP Server（FastMCP）
│   └── mcp_http_server.py        # 示例远程 MCP Server（FastMCP HTTP）
├── docs/                         # 设计文档
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

### Agent / Task / Multi-Agent Orchestrator

| 概念 | 职责 |
|------|------|
| Agent | 单 Agent：角色 + 工具 + 记忆 + 执行模式（chat / react / plan） |
| Task（PlanTask） | 计划步骤建模：id / 目标 / 类型 / 依赖，DAG 拓扑排序 + 并行调度 |
| MultiAgentOrchestrator | 多 Agent 编排：规划者拆解、执行者干活、检查者验收 |

### 记忆系统

| 层次 | 作用 |
|------|------|
| 短期记忆 | 对话/工具结果按 Token 预算淘汰，90% 触发 Map-Reduce 压缩，保留最近 3 轮 |
| 长期记忆 | 稳定事实 SQLite 持久化，归一化去重、关键词检索、global/仓库作用域隔离 |
| 工具记忆 | 记录每次工具调用的参数/结果/耗时，支撑循环内重复调用检测与拦截 |

---

## 路线图

- [x] 四种执行模式（chat / react / plan / multi）
- [x] ReAct 推理-行动循环（Thought / Action / Observation + 完成判断）
- [x] Plan & Execute（结构化计划 + 逐步执行 + 汇总）
- [x] Multi-Agent 协作（规划者/执行者/检查者 + 并行执行 + 审查重试）
- [x] 工具调用系统 + 按需装配
- [x] Function Calling 工具调用（双通道：FC 优先 + 文本协议 fallback）
- [x] MCP 协议化工具网关（本地工具 + MCP 远程工具共存）
- [x] 代码助手工具（read_file / write_file / list_files / find_files / grep_files / run_shell）
- [x] RAG 知识库（Chroma + bge 本地模型）
- [x] Code RAG 代码仓库检索（AST 索引 + BM25 + bge-m3 向量混合检索）
- [x] 长期记忆持久化（SQLite + 去重 + 关键词检索）
- [x] Agent 评测体系（确定性 checker + 沙箱隔离 + 逐条报告）
- [ ] Skill 技能包系统
- [ ] 长期记忆向量检索

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12 | 运行环境 |
| DeepSeek API | LLM 推理 |
| Chroma + bge-small-zh | RAG 知识库向量检索 |
| bge-m3 + BM25 + RRF | Code RAG 混合检索（代码仓库） |
| FastAPI + Uvicorn | REST API 服务 |
| Docker / Compose | 容器化部署 |
| 零框架依赖 | 编排引擎、工具、记忆、多 Agent 协作全部手写 |
