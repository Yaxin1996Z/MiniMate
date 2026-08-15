"""内置评测集 —— 覆盖 chat / react / plan / multi 四种模式

评测任务全部为本地确定性任务（不依赖网络），判定由确定性 checker 完成，
保证可复现；每条用例在独立临时沙箱执行，互不影响。
"""

from __future__ import annotations

from .case import EvalCase


BASIC_SUITE: list[EvalCase] = [
    # ----------------------------------------------------------
    # chat：单次 LLM 调用直接回答
    # ----------------------------------------------------------
    EvalCase(
        id="chat_001",
        name="概念问答：Python 装饰器",
        mode="chat",
        prompt="用一句话解释 Python 装饰器是什么",
        checker="output_contains",
        expected="函数",
        tags=["问答", "概念"],
    ),
    EvalCase(
        id="chat_002",
        name="概念问答：MCP 是什么",
        mode="chat",
        prompt="用两句话介绍 MCP（Model Context Protocol）是什么",
        checker="output_contains",
        expected="工具",
        tags=["问答", "概念"],
    ),
    EvalCase(
        id="chat_003",
        name="概念问答：冒泡排序思路",
        mode="chat",
        prompt="简述冒泡排序的基本思路",
        checker="output_contains",
        expected="交换",
        tags=["问答", "算法"],
    ),
    EvalCase(
        id="chat_004",
        name="翻译：Hello world",
        mode="chat",
        prompt="把 Hello world 翻译成中文，直接输出译文",
        checker="output_contains",
        expected="你好",
        tags=["问答", "翻译"],
    ),
    # ----------------------------------------------------------
    # react：Thought → Action → Observation 工具循环
    # ----------------------------------------------------------
    EvalCase(
        id="react_001",
        name="工具调用：创建文件并写入内容",
        mode="react",
        prompt="使用工具创建 hello.txt，内容为：你好，MiniMate！",
        checker="file_contains",
        expected="hello.txt|你好",
        tags=["工具", "文件写入"],
    ),
    EvalCase(
        id="react_002",
        name="工具调用：数学计算",
        mode="react",
        prompt="计算 15 乘以 4 再减去 3，只输出最终数字",
        checker="math_answer",
        expected="57",
        tags=["工具", "计算"],
    ),
    EvalCase(
        id="react_003",
        name="工具调用：读取文件并回答",
        mode="react",
        setup=[{"write": {"path": "data.txt", "content": "秘密数字是 42"}}],
        prompt="读取当前目录下的 data.txt，告诉我里面的秘密数字是多少",
        checker="output_contains",
        expected="42",
        tags=["工具", "文件读取"],
    ),
    EvalCase(
        id="react_004",
        name="工具调用：检索包含关键字的文件",
        mode="react",
        setup=[
            {"write": {"path": "a.txt", "content": "foo bar"}},
            {"write": {"path": "b.txt", "content": "CONFIG_TOKEN=abc123"}},
            {"write": {"path": "c.txt", "content": "hello world"}},
        ],
        prompt="在当前目录中找出包含 CONFIG_TOKEN 的文件，回答文件名",
        checker="output_contains",
        expected="b.txt",
        tags=["工具", "文件检索"],
    ),
    EvalCase(
        id="react_005",
        name="工具调用：列出目录内容",
        mode="react",
        setup=[
            {"mkdir": "project"},
            {"write": {"path": "project/main.py", "content": "print('main')"}},
            {"write": {"path": "project/utils.py", "content": "print('utils')"}},
        ],
        prompt="列出 project 目录下的所有文件名",
        checker="output_contains",
        expected="main.py",
        tags=["工具", "目录列举"],
    ),
    EvalCase(
        id="react_006",
        name="工具调用：写脚本并运行",
        mode="react",
        prompt="创建 fib.py 计算斐波那契数列前 8 项并打印，然后运行它",
        checker="command_exit0",
        expected="python fib.py",
        tags=["工具", "脚本执行"],
    ),
    # ----------------------------------------------------------
    # plan：生成计划 → 逐步执行 → 汇总
    # ----------------------------------------------------------
    EvalCase(
        id="plan_001",
        name="计划执行：脚本编写与运行",
        mode="plan",
        prompt=(
            "完成一个多步骤任务：先创建 stats.py，对列表 [3,1,4,1,5,9,2,6] "
            "求和并打印结果；然后运行 stats.py 验证输出"
        ),
        checker="command_exit0",
        expected="python stats.py",
        tags=["计划", "脚本执行"],
    ),
    EvalCase(
        id="plan_002",
        name="计划执行：多文件任务与验证",
        mode="plan",
        prompt=(
            "依次完成：1) 创建 README.md 写入 MiniMate 评测说明；"
            "2) 创建 CHANGELOG.md 写入 v0.1.0 发布；"
            "3) 验证两个文件都存在"
        ),
        checker="command_exit0",
        expected=(
            "python -c \"import os; "
            "assert os.path.exists('README.md') and os.path.exists('CHANGELOG.md'); "
            "print('ok')\""
        ),
        tags=["计划", "文件操作"],
    ),
    EvalCase(
        id="plan_003",
        name="计划执行：算法实现与测试",
        mode="plan",
        prompt=(
            "用 Python 实现二分查找保存为 binary_search.py，"
            "包含测试用例（含未找到的情况），运行测试验证"
        ),
        checker="command_exit0",
        expected="python binary_search.py",
        tags=["计划", "算法"],
    ),
    # ----------------------------------------------------------
    # multi：规划者拆解 → 多 Worker 并行 → 检查者验收
    # ----------------------------------------------------------
    EvalCase(
        id="multi_001",
        name="多 Agent：模块化小项目",
        mode="multi",
        prompt=(
            "创建一个小项目：1) utils.py 提供 add(a, b) 函数；"
            "2) main.py 导入 add 并打印 3+5 的结果；"
            "3) 运行 main.py 确认输出正确"
        ),
        checker="command_exit0",
        expected="python main.py",
        tags=["多Agent", "项目"],
    ),
    EvalCase(
        id="multi_002",
        name="多 Agent：拓扑排序算法与测试",
        mode="multi",
        prompt=(
            "创建 topo.py 实现 Kahn 拓扑排序算法，"
            "附测试用例（含环检测），运行验证通过"
        ),
        checker="command_exit0",
        expected="python topo.py",
        tags=["多Agent", "算法"],
    ),
    EvalCase(
        id="multi_003",
        name="多 Agent：并行创建配置文件",
        mode="multi",
        prompt=(
            "创建两个配置文件：config.yaml 内容为 database: minimate；"
            "notes.md 内容为 多 Agent 评测记录。"
            "完成后验证两个文件都存在且内容正确"
        ),
        checker="command_exit0",
        expected=(
            "python -c \"import os; "
            "c = open('config.yaml', encoding='utf-8').read(); "
            "n = open('notes.md', encoding='utf-8').read(); "
            "assert 'minimate' in c and '多 Agent' in n; "
            "print('ok')\""
        ),
        tags=["多Agent", "并行"],
    ),
]


_SUITES = {"basic": BASIC_SUITE}


def list_suites() -> list[str]:
    """可用评测集名称"""
    return sorted(_SUITES)


def get_suite(name: str = "basic") -> list[EvalCase]:
    """按名称获取评测集，未知名称抛出 ValueError"""
    if name not in _SUITES:
        raise ValueError(
            f"未知评测集：{name}，可用：{', '.join(list_suites())}"
        )
    return _SUITES[name]
