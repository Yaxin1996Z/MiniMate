"""
MiniMate CLI —— 工作/代码助手，支持三种 Agent 模式

用法：
  minimate "问题"                     # 默认 react 模式
  minimate "问题" --mode chat         # 纯问答（单次 LLM 调用）
  minimate "问题" --mode react        # ReAct 循环（工具调用）
  minimate "问题" --mode plan         # Plan & Execute
  minimate "问题" --kb-path ./docs    # 加载自定义知识库
  minimate --rebuild                  # 重建知识库索引
  minimate                            # 交互模式（多轮对话，短期记忆）
"""

import sys
import argparse
import os
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()

from minimate import __version__
from minimate.tools import (
    ToolExecutor,
    save_file,
    web_search,
    query_knowledge,
    read_file,
    list_files,
    write_file,
    run_shell,
    find_files,
    grep_files,
)
from minimate.memory import ResearchMemory
from minimate.orchestrator import Agent
from minimate.rag import get_knowledge_base
from minimate.colors import color
from minimate.config import get_mcp_servers
from minimate.mcp import McpToolAdapter


# 保持 MCP 连接引用，防止被 GC 回收
_mcp_adapters: list[McpToolAdapter] = []


def _load_mcp_tools(tools: ToolExecutor):
    """从 config.json 读取 MCP 配置，加载远程工具到执行器"""
    global _mcp_adapters
    servers = get_mcp_servers()
    if not servers:
        return
    for server in servers:
        name = server.get("name", "mcp")
        try:
            adapter = McpToolAdapter(
                server_name=name,
                command=server["command"],
                args=server.get("args", []),
                env=server.get("env"),
            )
            mcp_tools = adapter.connect()
            for t in mcp_tools:
                tools.register(t)
            _mcp_adapters.append(adapter)
            print(f"  MCP 已加载：{name}（{len(mcp_tools)} 个工具）")
        except Exception as e:
            print(f"  ⚠️ MCP 加载失败：{name} - {e}")


# ============================================================
# 交互模式界面
# ============================================================

BANNER = r"""
    __  ____       _ __  ___      __     
   /  |/  (_)___  (_)  |/  /___ _/ /____ 
  / /|_/ / / __ \/ / /|_/ / __ `/ __/ _ \
 / /  / / / / / / / /  / / /_/ / /_/  __/
/_/  /_/_/_/ /_/_/_/  /_/\__,_/\__/\___/ 
"""

HELP_TEXT = """
可用命令：
  /mode <chat|react|plan>   切换执行模式
  /memory                   查看当前会话记忆（短期）
  /clear                    清空会话记忆
  /help                     显示本帮助
  /quit 或 Ctrl+C           退出并清除记忆

直接输入问题即可开始对话，会话记忆保存在内存中，退出后自动清除。
"""


# ============================================================
# 执行入口
# ============================================================

def build_agent(
    memory: ResearchMemory,
    kb_path: str = "",
    max_steps: int = 8,
) -> Agent:
    """装配工具并创建 Agent（单次执行与交互模式共用）"""

    tools = ToolExecutor()
    tools.register(web_search)
    tools.register(save_file)
    tools.register(read_file)
    tools.register(list_files)
    tools.register(write_file)
    tools.register(run_shell)
    tools.register(find_files)
    tools.register(grep_files)

    # 知识库（非空才注册检索工具）
    kb = get_knowledge_base(repo_dir=kb_path)
    if kb.count() > 0:
        tools.register(query_knowledge)
        print(f"  知识库已加载：{kb.count()} 个片段")
    else:
        print("  知识库为空，未注册检索工具")

    # MCP 工具（config.json 配置了才加载）
    _load_mcp_tools(tools)

    return Agent(
        role="全能工作助手",
        goal="帮助用户高效完成任务：回答问题、检索信息、数学计算、整理输出",
        backstory=(
            "你是可靠的工作助手，既能直接回答问题，"
            "也能在需要时主动调用工具获取信息、进行计算、保存结果。"
        ),
        tools=tools,
        memory=memory,
        max_steps=max_steps,
    )


def run_query(
    question: str,
    mode: str = "react",
    kb_path: str = "",
    max_steps: int = 8,
) -> str:
    """以指定模式执行一次任务，返回最终答案"""

    memory = ResearchMemory()
    memory.add_user_message(question)
    agent = build_agent(memory, kb_path, max_steps)

    print(color.cyan(f"\n{'=' * 60}"))
    print(color.cyan(f"  MiniMate v{__version__}  [{mode} 模式]", bold=True))
    print(f"  问题：{question}")
    print(f"  启动时间：{datetime.now().strftime('%H:%M:%S')}")
    print(color.cyan(f"{'=' * 60}"))

    return agent.run(question, mode=mode)


# ============================================================
# 交互模式（REPL）
# ============================================================

def interactive(kb_path: str = "", max_steps: int = 8):
    """多轮对话窗口：短期记忆存于内存，Ctrl+C 退出并清除"""

    print(color.cyan(BANNER))
    print(color.cyan(f"  MiniMate v{__version__}", bold=True) + "  ·  工作/代码助手 Agent")
    print(f"  会话记忆为短期记忆（内存中），退出后自动清除")
    print(f"  输入 /help 查看命令，Ctrl+C 退出")

    memory = ResearchMemory()
    agent = build_agent(memory, kb_path, max_steps)
    mode = "react"
    print(color.green(f"\n  当前模式：{mode}", bold=True) + "（可用 /mode chat|plan 切换）")

    def handle_command(line: str) -> bool:
        """处理斜杠命令，返回 False 表示退出"""
        nonlocal mode
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()

        if cmd == "/help":
            print(color.yellow(HELP_TEXT))
        elif cmd in ("/quit", "/exit"):
            return False
        elif cmd == "/mode":
            if arg in ("chat", "react", "plan"):
                mode = arg
                print(color.green(f"  已切换到 {arg} 模式"))
            else:
                print(color.red("  用法：/mode chat|react|plan"))
        elif cmd == "/memory":
            ctx = memory.get_context()
            print(color.yellow(ctx) if ctx else color.yellow("  记忆为空"))
        elif cmd == "/clear":
            memory.clear()
            print(color.green("  会话记忆已清空"))
        else:
            print(color.red(f"  未知命令：{cmd}（输入 /help 查看帮助）"))
        return True

    try:
        while True:
            try:
                line = input(color.green("\n> ", bold=True)).strip()
            except EOFError:
                print(color.green("\n  再见！会话记忆已清除"))
                break

            if not line:
                continue

            if line.startswith("/"):
                if not handle_command(line):
                    print(color.green("  再见！会话记忆已清除"))
                    break
                continue

            memory.add_user_message(line)
            answer = agent.run(line, mode=mode)
            print(f"\n{answer}")
            memory.add_ai_message(answer)
    except KeyboardInterrupt:
        print(color.green("\n\n  再见！会话记忆已清除"))


# ============================================================
# CLI
# ============================================================

def cli():
    parser = argparse.ArgumentParser(
        description="MiniMate - 工作/代码助手 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
交互模式：
  minimate                                    # 进入多轮对话窗口
  minimate                                    # 会话中可用 /mode /memory /clear 等命令

示例：
  minimate "什么是 RAG"                            # 默认 react 模式
  minimate "什么是 RAG" --mode chat                # 纯问答
  minimate "帮我搜索 MCP 最新进展" --mode react     # ReAct 循环
  minimate "对比 RAG 和微调方案" --mode plan        # Plan & Execute
  minimate "Python 装饰器" --kb-path ./docs        # 加载自定义知识库
  minimate --rebuild                               # 重建知识库索引
  minimate                                        # 交互模式（多轮对话）
        """,
    )
    parser.add_argument("question", nargs="?", default="", help="任务/问题（留空进入交互模式）")
    parser.add_argument(
        "--mode",
        choices=["chat", "react", "plan"],
        default="react",
        help="Agent 执行模式：chat=纯问答, react=ReAct 循环, plan=Plan&Execute",
    )
    parser.add_argument("--kb-path", default="", help="知识库文档目录路径")
    parser.add_argument("--max-steps", type=int, default=8, help="ReAct 最大循环步数")
    parser.add_argument("--rebuild", action="store_true", help="强制重建知识库索引")
    parser.add_argument("--version", "-v", action="store_true", help="显示版本")

    args = parser.parse_args()

    if args.version:
        print(f"MiniMate v{__version__}")
        return

    if args.rebuild:
        from minimate.rag import get_knowledge_base
        kb = get_knowledge_base(repo_dir=args.kb_path)
        kb.rebuild()
        print(f"  知识库已重建，共 {kb.count()} 个片段")
        return

    if not args.question:
        interactive(kb_path=args.kb_path, max_steps=args.max_steps)
        return

    answer = run_query(
        args.question,
        mode=args.mode,
        kb_path=args.kb_path,
        max_steps=args.max_steps,
    )
    print(color.green(f"\n{'=' * 60}"))
    print(color.green("  最终答案：", bold=True))
    print(color.green(f"{'=' * 60}"))
    print(answer)


def main():
    cli()


if __name__ == "__main__":
    main()
