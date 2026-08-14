"""
MiniMate CLI —— 工作/代码助手，支持三种 Agent 模式

用法：
  minimate "问题"                     # 默认 react 模式
  minimate "问题" --mode chat         # 纯问答（单次 LLM 调用）
  minimate "问题" --mode react        # ReAct 循环（工具调用）
  minimate "问题" --mode plan         # Plan & Execute
  minimate "问题" --mode multi        # Multi-Agent（规划者/执行者/检查者协作）
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
from minimate.tools import Tool, ToolExecutor, register_all_tools
from minimate.memory import MemoryManager
from minimate.agent import Agent, MultiAgentOrchestrator
from minimate.rag import get_knowledge_base
from minimate.colors import color
from minimate.config import get_mcp_servers
from minimate.mcp import McpToolAdapter
from minimate.logging import log_path
from minimate.coderag import search_code


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
                transport=server.get("transport", "stdio"),
                command=server.get("command"),
                args=server.get("args", []),
                env=server.get("env"),
                url=server.get("url"),
                headers=server.get("headers"),
                oauth=server.get("oauth"),
            )
            mcp_tools = adapter.connect()
            for t in mcp_tools:
                tools.register(t)
            _mcp_adapters.append(adapter)
            print(f"  MCP 已加载：{name}（{len(mcp_tools)} 个工具）")
        except Exception as e:
            print(f"  ⚠️ MCP 加载失败：{name} - {e}")


def _mcp_status_report() -> str:
    """生成 MCP 服务器连接状态报告"""
    if not _mcp_adapters:
        return "  未配置 MCP 服务器（config.json 的 mcp.servers）"
    lines = ["  MCP 服务器状态："]
    icons = {
        "connected": "✅",
        "connecting": "⏳",
        "pending": "⏸",
        "failed": "❌",
        "closed": "⏹",
    }
    for a in _mcp_adapters:
        icon = icons.get(a.status, "❓")
        base = f"    {icon} {a.server_name} [{a.transport}] {a.status}"
        if a.status == "connected":
            base += f" · {a.tool_count} 个工具 · {a.connected_at} 连接"
        if a.error:
            base += f"\n      错误：{a.error}"
        lines.append(base)
    return "\n".join(lines)


def _stats_report() -> str:
    """生成 LLM Token 用量统计报告"""
    from minimate.llm import get_stats

    s = get_stats()
    total = s["prompt_tokens"] + s["completion_tokens"]
    return (
        "  LLM 用量统计：\n"
        f"    调用次数：{s['calls']}\n"
        f"    prompt tokens：{s['prompt_tokens']}\n"
        f"    completion tokens：{s['completion_tokens']}\n"
        f"    合计：{total} tokens"
    )


def _memory_report(memory) -> str:
    """生成记忆系统报告（统计 + 长期事实 + 摘要 + 短期上下文）"""
    stats = memory.stats()
    lines = [
        f"  记忆统计：短期 {stats['short_term_items']} 条"
        f"（{stats['short_term_tokens']} tokens）· "
        f"摘要 {stats['summaries']} 条 · "
        f"长期事实 {stats['long_term_facts']} 条",
    ]
    facts = memory.long_term.items
    if facts:
        lines.append("\n  【长期记忆】")
    lines.extend(f"    - [{f.id}] {f.content[:100]}" for f in facts[-5:])
    summaries = memory.short_term.get_summaries()
    if summaries:
        lines.append("\n  【历史摘要】")
        lines.append("    " + summaries.replace("\n", "\n    "))
    ctx = memory.short_term.get_context()
    if ctx:
        lines.append("\n  【最近对话】")
        lines.append("    " + ctx[:400].replace("\n", "\n    "))
    return "\n".join(lines)


def _remember_repo_index(memory, name: str, chunks: int | None = None):
    """把仓库索引状态写入长期记忆（替换旧的同仓库条目），让 LLM 优先用 search_code 检索代码"""
    marker = f"代码仓库 {name}"
    for item in memory.list_long_term():
        if item.content.startswith("项目信息：") and marker in item.content:
            memory.delete_long_term(item.id)
    if chunks is None:
        fact = f"项目信息：代码仓库 {name} 已配置，查询该仓库代码请使用 search_code 工具"
    else:
        fact = (
            f"项目信息：代码仓库 {name} 已建立索引（{chunks} 个代码块），"
            "查询该仓库代码请直接使用 search_code 工具检索"
        )
    memory.store_fact(fact, source="coderag", project=name)


def _memory_with_repos() -> MemoryManager:
    """创建带已配置仓库登记的记忆管理器（用于检索时按仓库过滤）"""
    memory = MemoryManager()
    try:
        from minimate.coderag import CodeRAGManager

        memory.set_repos(list(CodeRAGManager().list_repos().keys()))
    except Exception:
        pass
    return memory


def _repos_command(args: str, memory=None) -> str:
    """处理 /repos 子命令：add / list / index / update / search"""
    from minimate.coderag import CodeRAGManager

    parts = (args or "").split()
    mgr = CodeRAGManager()

    if not parts or parts[0] == "list":
        repos = mgr.list_repos()
        if not repos:
            return "未配置代码仓库。用法：/repos add <名称> <路径|URL>"
        return "已配置代码仓库：\n" + "\n".join(
            f"  - {k}: {v}" for k, v in repos.items()
        )

    action = parts[0]
    if action == "add" and len(parts) >= 3:
        name, source = parts[1], parts[2]
        mgr.add_repo(name, source)
        if memory:
            _remember_repo_index(memory, name)
        return f"已配置仓库 {name} -> {source}\n请执行 /repos index {name} 构建索引"

    if action == "index" and len(parts) >= 2:
        name = parts[1]
        try:
            info = mgr.index(name)
        except Exception as e:
            return f"[索引失败] {e}"
        if memory:
            _remember_repo_index(memory, name, chunks=info["chunks"])
        return (
            f"索引完成：{info['chunks']} 个代码块，{info['relations']} 条关系\n"
            f"DB: {info['db']}"
        )

    if action == "update" and len(parts) >= 2:
        name = parts[1]
        try:
            info = mgr.update(name)
        except Exception as e:
            return f"[更新失败] {e}"
        if memory:
            _remember_repo_index(memory, name, chunks=info["chunks"])
        return (
            f"更新完成：{info['chunks']} 个代码块，{info['relations']} 条关系\n"
            f"DB: {info['db']}"
        )

    if action == "search" and len(parts) >= 3:
        name, query = parts[1], " ".join(parts[2:])
        try:
            results = mgr.search(name, query, top_k=5)
        except Exception as e:
            return f"[检索错误] {e}"
        if not results:
            return f"未找到相关代码（仓库：{name}）"
        lines = [f"相关代码（{name}）："]
        for c in results:
            lines.append(
                f"- [{c['granularity']}] {c['name']}  "
                f"{c['file_path']}:{c['start_line']}（相似度 {c.get('score', 0)}）"
            )
        return "\n".join(lines)

    return (
        "用法：\n"
        "  /repos add <名称> <路径|URL>    配置代码仓库\n"
        "  /repos list                    列出已配置仓库\n"
        "  /repos index <名称>            构建 AST 索引\n"
        "  /repos update <名称>           拉取更新并重建索引（git 仓库先 pull）\n"
        "  /repos search <名称> <查询>    自然语言检索代码"
    )


def _save_fact(memory: MemoryManager):
    """长期记忆写入工具：scope=global 用户偏好（跨项目）；scope=repo 仓库事实"""
    from typing import Literal

    def save_fact(
        content: str,
        scope: Literal["global", "repo"] = "repo",
        repo: str = "",
    ) -> str:
        if not content or len(content.strip()) < 4:
            return "[错误] 内容太短，未保存"
        internal_scope = "global" if scope == "global" else "project"
        ok = memory.store_fact(
            content.strip(),
            scope=internal_scope,
            project=repo.strip() or None,
            source="agent",
        )
        return "已保存到长期记忆" if ok else "内容重复或无效，跳过"

    return save_fact


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
  /mode <chat|react|plan|multi>   切换执行模式
  /mcp                      查看 MCP 服务器连接状态
  /stats                    查看 LLM Token 用量统计
  /repos                    配置/索引/更新/检索代码仓库（Code RAG）
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
    memory: MemoryManager,
    kb_path: str = "",
    max_steps: int = 8,
) -> Agent:
    """装配工具并创建 Agent（单次执行与交互模式共用）"""

    tools = ToolExecutor()
    # 注册全部内置工具（按类分组：文件/Shell/Web/RAG，见 tools/registry.py）
    register_all_tools(tools)
    # Code RAG 检索工具（配置了代码仓库后可用）
    tools.register(search_code)
    # 长期记忆写入工具：模型在对话中主动保存跨会话事实
    tools.register(
        Tool(
            name="save_fact",
            description=(
                "把一条跨会话稳定的信息保存到长期记忆。"
                "scope=global 用于用户偏好/习惯（跨项目通用）；"
                "scope=repo 用于代码仓库/项目相关事实（可指定 repo 仓库名）。"
                "只保存值得长期记住的稳定信息（偏好、约定、项目技术栈等），"
                "不要保存临时任务、一次性指令或当前对话过程。"
            ),
            func=_save_fact(memory),
        )
    )

    # 知识库（用于提示加载状态）
    kb = get_knowledge_base(repo_dir=kb_path)
    if kb.count() > 0:
        print(f"  知识库已加载：{kb.count()} 个片段")
    else:
        print("  知识库为空（query_knowledge 将返回空结果）")

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
        tool_memory=memory.tool_memory,
    )


def _run_agent_query(
    question: str,
    mode: str,
    agent: Agent,
    memory: MemoryManager,
    max_steps: int = 8,
) -> str:
    """按模式执行一次任务：multi 走多 Agent 编排器，其余走单 Agent"""
    if mode == "multi":
        orchestrator = MultiAgentOrchestrator(
            tools=agent.tools,
            memory=memory,
            max_steps=max_steps,
        )
        return orchestrator.run(question)
    return agent.run(question, mode=mode)


def run_query(
    question: str,
    mode: str = "react",
    kb_path: str = "",
    max_steps: int = 8,
) -> str:
    """以指定模式执行一次任务，返回最终答案"""

    memory = _memory_with_repos()
    memory.add_user_message(question)
    agent = build_agent(memory, kb_path, max_steps)

    print(color.cyan(f"\n{'=' * 60}"))
    print(color.cyan(f"  MiniMate v{__version__}  [{mode} 模式]", bold=True))
    print(f"  问题：{question}")
    print(f"  启动时间：{datetime.now().strftime('%H:%M:%S')}")
    print(color.cyan(f"{'=' * 60}"))

    answer = _run_agent_query(question, mode, agent, memory, max_steps)
    memory.save()  # 长期事实持久化，跨会话保留
    return answer


# ============================================================
# 交互输入：↑/↓ 历史导航
# ============================================================

def _input_line(history: list[str], prompt: str = "> ") -> str:
    """带历史记录（↑/↓ 翻看最近发送的消息）的输入。

    Windows 交互终端用 msvcrt 自绘编辑；其他平台优先 readline（自带历史）；
    非终端（管道/测试）回退普通 input()。
    """
    if os.name == "nt" and sys.stdin.isatty() and sys.stdout.isatty():
        return _input_windows(history, prompt)
    try:
        import readline  # noqa: F401  # Unix 原生；Windows 安装 pyreadline3 后也可用
        readline.set_history_length(100)
    except ImportError:
        pass
    return input("\n" + prompt)


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 等宽字符按 2 列计，其余按 1 列"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _input_windows(history: list[str], prompt: str) -> str:
    """Windows 控制台输入：支持 ↑/↓ 历史、←/→ 移动光标、退格/删除、Home/End、Ctrl+C"""
    import msvcrt

    sys.stdout.write("\n" + prompt)
    sys.stdout.flush()
    buf: list[str] = []
    cursor = 0
    hist_index = len(history)
    draft = ""
    prev_len = 0

    def redraw():
        nonlocal prev_len
        line = "".join(buf)
        width = _display_width(line)
        sys.stdout.write("\r" + prompt + line + " " * max(0, prev_len - width))
        sys.stdout.write("\r" + prompt + line)
        sys.stdout.write("\b" * _display_width(line[cursor:]))
        prev_len = width
        sys.stdout.flush()

    while True:
        ch = msvcrt.getwch()
        if ch == "\r":
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(buf)
        if ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if ch == "\x1a":  # Ctrl+Z → EOF
            raise EOFError
        if ch in ("\x00", "\xe0"):  # 功能键（方向键等）
            key = msvcrt.getwch()
            if key == "H":  # ↑ 历史上一条
                if hist_index > 0:
                    if hist_index == len(history):
                        draft = "".join(buf)
                    hist_index -= 1
                    buf = list(history[hist_index])
                    cursor = len(buf)
            elif key == "P":  # ↓ 历史下一条
                if hist_index < len(history):
                    hist_index += 1
                    if hist_index == len(history):
                        buf = list(draft)
                    else:
                        buf = list(history[hist_index])
                    cursor = len(buf)
            elif key == "K":  # ←
                cursor = max(0, cursor - 1)
            elif key == "M":  # →
                cursor = min(len(buf), cursor + 1)
            elif key == "G":  # Home
                cursor = 0
            elif key == "O":  # End
                cursor = len(buf)
            elif key == "S":  # Delete
                if cursor < len(buf):
                    del buf[cursor]
            redraw()
            continue
        if ch == "\x08":  # Backspace
            if cursor > 0:
                del buf[cursor - 1]
                cursor -= 1
        else:
            buf.insert(cursor, ch)
            cursor += 1
        redraw()


# ============================================================
# 交互模式（REPL）
# ============================================================

def interactive(kb_path: str = "", max_steps: int = 8):
    """多轮对话窗口：短期记忆存于内存，Ctrl+C 退出并清除"""

    print(color.cyan(BANNER))
    print(color.cyan(f"  MiniMate v{__version__}", bold=True) + "  ·  工作/代码助手 Agent")
    print(f"  短期记忆自动管理（Token 预算 + 压缩）；长期记忆持久化到 ~/.minimate/memory.db")
    print(f"  输入 /help 查看命令，Ctrl+C 退出")

    memory = _memory_with_repos()
    agent = build_agent(memory, kb_path, max_steps)
    mode = "react"
    history: list[str] = []
    print(color.green(f"\n  当前模式：{mode}", bold=True) + "（可用 /mode chat|plan|multi 切换）")

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
            if arg in ("chat", "react", "plan", "multi"):
                mode = arg
                print(color.green(f"  已切换到 {arg} 模式"))
            else:
                print(color.red("  用法：/mode chat|react|plan|multi"))
        elif cmd == "/mcp":
            print(color.cyan(_mcp_status_report()))
        elif cmd == "/stats":
            print(color.cyan(_stats_report()))
        elif cmd == "/repos":
            print(color.cyan(_repos_command(arg, memory)))
        elif cmd == "/memory":
            if arg.startswith("del "):
                entry_id = arg.split(maxsplit=1)[1].strip()
                if memory.delete_long_term(entry_id):
                    print(color.green(f"  已删除长期事实 #{entry_id}"))
                else:
                    print(color.red(f"  未找到 id={entry_id} 的长期事实"))
            else:
                print(color.yellow(_memory_report(memory)))
        elif cmd == "/clear":
            memory.clear()
            print(color.green("  会话记忆已清空"))
        else:
            print(color.red(f"  未知命令：{cmd}（输入 /help 查看帮助）"))
        return True

    try:
        while True:
            try:
                line = _input_line(history, prompt=color.green("> ", bold=True)).strip()
            except EOFError:
                memory.save()
                print(color.green("\n  再见！会话记忆已清除"))
                break

            if not line:
                continue

            if len(history) >= 100:
                history.pop(0)
            history.append(line)

            if line.startswith("/"):
                if not handle_command(line):
                    memory.save()
                    print(color.green("  再见！会话记忆已清除"))
                    break
                continue

            memory.add_user_message(line)
            answer = _run_agent_query(line, mode, agent, memory, max_steps)
            memory.add_ai_message(answer)
    except KeyboardInterrupt:
        memory.save()
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
  minimate "重构工具注册模块" --mode multi          # Multi-Agent 协作
  minimate "Python 装饰器" --kb-path ./docs        # 加载自定义知识库
  minimate --rebuild                               # 重建知识库索引
  minimate                                        # 交互模式（多轮对话）
        """,
    )
    parser.add_argument("question", nargs="?", default="", help="任务/问题（留空进入交互模式）")
    parser.add_argument(
        "--mode",
        choices=["chat", "react", "plan", "multi"],
        default="react",
        help="Agent 执行模式：chat=纯问答, react=ReAct 循环, plan=Plan&Execute, multi=多 Agent 协作",
    )
    parser.add_argument("--kb-path", default="", help="知识库文档目录路径")
    parser.add_argument("--max-steps", type=int, default=8, help="ReAct 最大循环步数")
    parser.add_argument("--rebuild", action="store_true", help="强制重建知识库索引")
    parser.add_argument("--version", "-v", action="store_true", help="显示版本")
    parser.add_argument("--verbose", action="store_true", help="开启控制台日志（默认仅写入文件）")

    args = parser.parse_args()

    if args.verbose:
        from minimate.logging import setup_logger
        setup_logger(console=True)

    print(f"  日志：{log_path()}")

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
    print(f"\n{_stats_report()}")


def main():
    cli()


if __name__ == "__main__":
    main()
