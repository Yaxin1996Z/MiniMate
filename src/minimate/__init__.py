"""MiniMate - 工作/代码助手 Agent 系统

一个可扩展的 Agent 框架，支持三种执行模式：
  chat  - 纯问答：单次 LLM 调用直接返回
  react - ReAct 循环：Thought → Action → Observation → Final Answer
  plan  - Plan & Execute：结构化计划 → 逐步执行 → 汇总

核心能力：工具调用、记忆管理、RAG 知识库检索、多 Agent 编排

启动：minimate "你的问题"
"""

__version__ = "0.1.0"
